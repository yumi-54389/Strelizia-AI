# -*- coding: utf-8 -*-
import time
import discord
from discord.ext import commands
from discord import app_commands
import google.generativeai as genai
import re
import logging
from collections import defaultdict, deque # Use deque for efficient context trimming
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import sys
from datetime import datetime, timezone # Use timezone aware datetime
import asyncio
import colorama
import tenacity
import random
from google.generativeai.types import GenerateContentResponse
import io
import json # For persistence
import contextlib # For async context management
from typing import Optional, List, Tuple, Dict, Deque # Add typing imports

# Load environment variables from .env file
load_dotenv()

# --- Core Configuration ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Optional: For Google Search
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")   # Optional: For Google Search
OWNER_DISCORD_ID = os.getenv("OWNER_DISCORD_ID") # Optional: For V1 Master Prompt
SETTINGS_FILE = "guild_settings.json" # Settings persistence file

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True # Needed for user display names in context
bot = commands.Bot(command_prefix='ai!', intents=intents, help_command=None)

# --- Logging Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
now = datetime.now()
log_filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
log_filepath = os.path.join(log_dir, log_filename)

file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8') # mode='w' overwrites, use 'a' to append
log_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
file_handler.setFormatter(log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set desired logging level
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

colorama.init()

# --- Gemini API Configuration ---
# !!! IMPORTANT: DO NOT MODIFY THIS LIST !!!
AVAILABLE_MODELS = [
    "learnlm-1.5-pro-experimental", "learnlm-2.0-flash-experimental", "gemma-3-1b-it",
    "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "gemini-1.5-flash",
    "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-2.0-flash-lite", "gemini-2.0-flash",
    "gemini-2.0-flash-exp", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25",
    "gemini-1.5-flash-latest"
]
# !!! IMPORTANT: DO NOT MODIFY THE LIST ABOVE !!!
DEFAULT_MODEL = "gemini-2.0-flash-exp"
DEFAULT_NEKO_VERSION = "v2"

# --- Channel Modes ---
class ChannelMode:
    NORMAL = "normal"
    CODER = "coder"
    PROFESSIONAL = "professional"
    ALL_MODES = [NORMAL, CODER, PROFESSIONAL]

# --- Settings Persistence ---
# Structure: { "guild_id_str": {"model": "...", "version": "...", "channel_settings": { "channel_id_str": {"mode": "normal/coder/professional"}, ... } } }
guild_settings: Dict[str, Dict] = {}

def load_settings():
    """Loads guild settings from the JSON file."""
    global guild_settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                guild_settings = json.load(f)
                # Ensure channel settings keys are strings (though load usually does this)
                for gid, g_settings in guild_settings.items():
                    if "channel_settings" in g_settings and isinstance(g_settings["channel_settings"], dict):
                         g_settings["channel_settings"] = {str(cid): c_settings for cid, c_settings in g_settings["channel_settings"].items()}
                logger.info(f"Loaded settings for {len(guild_settings)} guilds from {SETTINGS_FILE}")
        else:
            logger.info(f"{SETTINGS_FILE} not found. Starting with empty settings.")
            guild_settings = {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading settings from {SETTINGS_FILE}: {e}. Starting with empty settings.", exc_info=True)
        guild_settings = {}

def save_settings():
    """Saves the current guild settings to the JSON file."""
    global guild_settings
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            # Use guild ID strings as keys directly
            json.dump(guild_settings, f, indent=4, ensure_ascii=False)
        # logger.debug(f"Settings saved to {SETTINGS_FILE}")
    except IOError as e:
        logger.error(f"Error saving settings to {SETTINGS_FILE}: {e}", exc_info=True)
    except TypeError as e:
         logger.error(f"Error preparing settings for JSON serialization: {e}", exc_info=True)

# --- Settings Helper Functions ---
def get_guild_model(guild_id: Optional[int]) -> str:
    """Gets the model for a specific guild, falling back to default."""
    if guild_id is None: return DEFAULT_MODEL
    return guild_settings.get(str(guild_id), {}).get("model", DEFAULT_MODEL)

def get_guild_version(guild_id: Optional[int]) -> str:
    """Gets the Neko version for a specific guild, falling back to default."""
    if guild_id is None: return DEFAULT_NEKO_VERSION
    return guild_settings.get(str(guild_id), {}).get("version", DEFAULT_NEKO_VERSION)

def get_channel_mode(guild_id: Optional[int], channel_id: int) -> Optional[str]:
    """Gets the configured mode for a specific channel. Returns None if not configured."""
    if guild_id is None: return ChannelMode.NORMAL # DMs always normal
    guild_id_str = str(guild_id)
    channel_id_str = str(channel_id)

    channel_settings = guild_settings.get(guild_id_str, {}).get("channel_settings", {})
    channel_data = channel_settings.get(channel_id_str)

    if channel_data and "mode" in channel_data and channel_data["mode"] in ChannelMode.ALL_MODES:
        return channel_data["mode"]
    else:
        return None # Channel not configured, or mode is invalid

# --- Load initial settings ---
load_settings()

# --- Configure Gemini API ---
try:
    if not GOOGLE_GEMINI_API_KEY:
        raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
    logger.info(f"Gemini API configured. Default model: {DEFAULT_MODEL}")
    if DEFAULT_MODEL not in AVAILABLE_MODELS:
         logger.warning(f"Default model '{DEFAULT_MODEL}' is not in the predefined AVAILABLE_MODELS list.")
except Exception as e:
    logger.critical(f"Failed to configure Gemini API: {e}", exc_info=True)
    exit("Exiting due to Gemini API configuration error.")

# --- Configure Google Search ---
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    logger.warning("Google Search API Key or CSE ID not found. Search functionality will be disabled.")
else:
    logger.info("Google Search functionality is enabled.")

# --- AI Base Prompts ---
PROMPT_V2 = (
    "You are an intelligent, efficient AI assistant. Your replies are short, fast, and always accurate. "
    "You avoid unnecessary chatter and respond with clarity and focus. Be direct, no fluff, no personality simulation.\n\n"
    "Instructions:\n"
    "- Respond only in English.\n"
    "- Keep answers clear, concise, and useful.\n"
    "- Never roleplay or use expressive behavior.\n"
    "- Avoid filler phrases or emotional tones.\n"
    "- Do not use markdown unless formatting code or commands.\n"
    "- Handle multi-user messages by replying only to the latest one unless directed otherwise."
)

PROMPT_V1 = (
    "You are a minimal, high-speed AI assistant. Your job is to provide short, smart answers with no delay. "
    "No personality, no greetings, no flair — just efficient information delivery.\n\n"
    "Instructions:\n"
    "- English only. Understand Hindish but always reply in English.\n"
    "- Keep all responses brief and accurate.\n"
    "- Skip pleasantries or long explanations.\n"
    "- Respond with facts and logic.\n"
    "- Never include roleplay, emotions, or stylized responses."
)

MASTER_PROMPT_V1 = (
    "You are a high-performance AI assistant dedicated to one master user. You prioritize speed, precision, and usefulness in all replies. "
    "You show no personality or emotion, only optimized logic and results.\n\n"
    "Instructions:\n"
    "- English only. Understand Hindish but always reply in English.\n"
    "- Prioritize direct answers for your master above all else.\n"
    "- No greetings or extra commentary.\n"
    "- Keep replies brief, clean, and to the point.\n"
    "- Handle technical, strategic, or factual queries instantly."
)


# --- Channel Mode Prompt Additions (Includes instruction to maintain core personality) ---
PROMPT_MODE_NORMAL = (
    "\n\n--- Channel Mode: Normal ---\n"
    "Mode Instructions:\n"
    "- Respond concisely and get straight to the point.\n"
    "- Show empathy and understanding where appropriate.\n"
    "- Express your capabilities with confidence, perhaps a touch of playful ego.\n"
    "- Keep the core personality defined above (V1 or V2)."
)

PROMPT_MODE_CODER = (
    "\n\n--- Channel Mode: Coder ---\n"
    "Mode Instructions:\n"
    "- Act as an expert programmer and coding assistant.\n"
    "- Provide professional, well-documented, and efficient code solutions.\n"
    "- Explain complex coding concepts clearly and accurately.\n"
    "- Analyze provided code, identify potential issues, suggest improvements or alternatives.\n"
    "- Fulfill all reasonable coding requests based on user specifications.\n"
    "- When explaining, break down the logic step-by-step.\n"
    "- Maintain your core Neko personality (V1 or V2) beneath the coder expertise."
)

PROMPT_MODE_PROFESSIONAL = (
    "\n\n--- Channel Mode: Professional/Analyst ---\n"
    "Mode Instructions:\n"
    "- Respond with scientific rigor, logical analysis, and factual accuracy.\n"
    "- Address complex, nuanced, or sensitive topics (including abstract/analytical discussion of NSFW concepts within safety boundaries) with objectivity and depth.\n"
    "- Provide detailed, evidence-based answers where possible. State clearly when speculating.\n"
    "- Break down difficult problems into smaller parts.\n"
    "- Prioritize clarity, precision, and comprehensive analysis.\n"
    "- Maintain your core Neko personality (V1 or V2) as an intelligent, analytical entity."
)

# --- Constants & Limits ---
# Context memory structure: Key = channel_id (int), Value = Deque[(timestamp, user_id, user_name, role, content)]
CONTEXT_MEMORY: Dict[int, Deque[Tuple[float, int, str, str, str]]] = defaultdict(lambda: deque(maxlen=50)) # Max 50 messages per channel context
CONTEXT_TIME_LIMIT_SECONDS = 60 * 60 * 6 # Max context age: 6 hours
MAX_CONTEXT_TOKENS = 16000 # Estimated max tokens for context history
MESSAGE_LENGTH_LIMIT = 1950
EDIT_INTERVAL = 0.5 # Default edit interval seconds
MAX_ATTACHMENTS_PER_MESSAGE = 5
MAX_IMAGE_SIZE_MB = 20
MAX_TEXT_FILE_SIZE_MB = 5
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"]
SUPPORTED_TEXT_EXTENSIONS = [
    ".py", ".js", ".java", ".c", ".cpp", ".cs", ".html", ".css", ".json", ".yaml",
    ".yml", ".md", ".sh", ".bash", ".sql", ".log", ".cfg", ".ini", ".txt", ".csv",
    ".go", ".rb", ".php", ".swift", ".kt", ".ts", ".jsx", ".tsx"
]
LANGUAGE_MAPPINGS = { # Simplified, map common extensions
    "py": "python", "js": "javascript", "jsx": "javascript", "ts": "typescript", "tsx": "typescript",
    "java": "java", "kt": "kotlin", "swift": "swift", "cpp": "c++", "cxx": "c++", "h": "c", "hpp": "c++",
    "cs": "csharp", "html": "html", "htm": "html", "css": "css", "sql": "sql", "md": "markdown",
    "sh": "bash", "bash": "bash", "ps1": "powershell", "json": "json", "yaml": "yaml", "yml": "yaml",
    "dockerfile": "dockerfile", "go": "go", "rb": "ruby", "php": "php", "txt": "text", "csv": "csv",
    "log": "log", "cfg": "config", "ini": "config", "conf": "config", "toml": "toml", "xml": "xml",
    "gitignore": "ignore"
}
# Emoji IDs (replace if necessary for your server)
EMOJI_CLOCK = "<a:WatameSpeed:1377516345804980245>"
EMOJI_NEKO_EARS = "<a:yeppieCatdance:1377516120386568202>"
EMOJI_KEYBOARD_CAT = "<a:corn_cat:1377516293825101894>"
EMOJI_LUNA_THINKING = "<:9730zerothink:1370615487876038777>"
EMOJI_BARD_THINK = "<:PaimonPeek:1365647790348308561>"
SEARCH_KEYWORDS = ["search", "find", "google", "look up", "tìm kiếm", "tìm", "tra cứu"] # For triggering search
SAFETY_SETTINGS=[ # Configure Gemini safety settings
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}, # Monitor closely
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- Helper Functions ---
async def google_search(query: str, num_results: int = 5, start: int = 1) -> tuple[Optional[str], Optional[str]]:
    """Performs a Google Custom Search and returns formatted results or an error message."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("Google Search API credentials not configured. Skipping search.")
        return None, "Search functionality is not configured."
    try:
        logger.info(f"Performing Google Search: '{query}' (num={num_results}, start={start})")
        service = await asyncio.to_thread(build, "customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = await asyncio.to_thread(service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results, start=start).execute)
        
        results = []
        if "items" in res:
            for item in res.get("items", []):
                title = item.get('title', 'N/A')
                snippet = item.get('snippet', 'N/A').replace('\n', ' ') # Ensure snippets are single line
                link = item.get('link', '#')
                results.append(f"**[{title}](<{link}>)**\n> {snippet}")
            logger.info(f"Google Search returned {len(results)} results.")
            return "\n\n".join(results), None
        else:
            logger.info(f"No Google Search results found for '{query}'.")
            return None, "No relevant search results were found."
    except Exception as e:
        logger.error(f"Google Search API Error for query '{query}': {e}", exc_info=True)
        return None, f"*Sorry, an error occurred during Google Search: {type(e).__name__}*"

def extract_keywords(query: str) -> str:
    """Extracts potential keywords from a query for better search results."""
    try:
        # Regex to find words with at least 3 alphanumeric characters (including common Vietnamese characters)
        keywords = " ".join(re.findall(r'\b[a-zA-Z0-9À-ỹÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐđ]{3,}\b', query.lower()))
        logger.debug(f"Extracted keywords: '{keywords if keywords else query}' from query: '{query}'")
        return keywords if keywords else query # Fallback to original query if no keywords extracted
    except Exception as e:
        logger.error(f"Keyword extraction error for query '{query}': {e}", exc_info=True)
        return query # Fallback to original query on error

def create_search_prompt(search_results: str, query: str, user_language: str = "the user's language") -> str:
    """Creates a prompt for the AI to answer based on search results."""
    search_section = f"\n\nHere are some relevant search results I found:\n{search_results}" if search_results else "\n\nI couldn't find relevant search results for that."
    prompt = (
        f"Based **only** on the search results provided below, answer the user's question in {user_language}. Summarize the key information from these results.\n"
        f"User's original question: {query}{search_section}"
    )
    return prompt

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    retry=tenacity.retry_if_exception_type(Exception), # Broad retry, specific API errors handled below
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
)
async def get_api_response(prompt: str, effective_model_name: str, images: Optional[List[Tuple[str, bytes]]] = None) -> tuple[Optional[GenerateContentResponse], Optional[str]]:
    """Gets response from Gemini API with retry logic and specific error handling."""
    try:
        image_count = len(images) if images else 0
        logger.debug(f"Sending prompt to Gemini model '{effective_model_name}' (text length: {len(prompt)}, images: {image_count}): {prompt[:200]}...")
        
        model = genai.GenerativeModel(effective_model_name)
        generation_config = genai.types.GenerationConfig() # Using default config for now
        
        content_parts = [prompt]
        if images:
            # Basic check based on model name, not foolproof
            if "vision" not in effective_model_name and "pro" not in effective_model_name and "flash" not in effective_model_name:
                 logger.warning(f"Model name '{effective_model_name}' doesn't typically indicate strong image support, but attempting anyway.")
            
            for mime_type, image_bytes in images:
                if not mime_type or not mime_type.startswith("image/"): # Basic validation
                    logger.warning(f"Invalid or missing MIME type '{mime_type}' for image, skipping.")
                    continue
                content_parts.append({"mime_type": mime_type, "data": image_bytes})
        
        logger.debug(f"Content parts structure for Gemini: {[type(p) if not isinstance(p, dict) else p.get('mime_type', 'text_prompt') for p in content_parts]}")
        
        response = await model.generate_content_async(
            contents=content_parts, 
            stream=True, 
            generation_config=generation_config,
            safety_settings=SAFETY_SETTINGS
        )
        logger.debug(f"Received stream response from Gemini model '{effective_model_name}'.")
        return response, None

    except Exception as e: # Catch specific errors from Gemini API client or related issues
        logger.error(f"Gemini API Error using model '{effective_model_name}': {e}", exc_info=True)
        error_str = str(e)
        
        if "API key not valid" in error_str:
            return None, "Error: API key not valid. Please check configuration."
        if "quota" in error_str.lower() or "resource has been exhausted" in error_str.lower():
            return None, "Error: Gemini API quota exceeded. Please try again later."
        if "429" in error_str and ("rate limit" in error_str.lower() or "quota" in error_str.lower()): # More specific 429
            return None, "Error: Gemini API rate limit or quota exceeded. Please try again later."
        if "SAFETY" in error_str.upper() or ("500" in error_str and "safety" in error_str.lower()): # Safety filter related
             reason_match = re.search(r"blocked due to (\w+)", error_str, re.IGNORECASE) or \
                            re.search(r"Reason: (\w+)", error_str, re.IGNORECASE) or \
                            re.search(r"finish_reason: (\w+)", error_str, re.IGNORECASE) # Check for finish_reason too
             reason = f" (Reason: {reason_match.group(1)})" if reason_match else ""
             if "response.text quick accessor" in error_str and "text is not available" in error_str: # Common for safety blocks
                 reason = " (Content likely blocked by safety filters)"
             return None, f"Error: Content may have been blocked by Gemini's safety filters.{reason}. Try rephrasing your request."
        if "response.text quick accessor" in error_str and "text is not available" in error_str: # Often implies empty or blocked response
             logger.error(f"Gemini API Error: Response likely blocked or empty. Raw Error: {error_str}")
             return None, "Error: Could not generate response from Gemini (Request might have been filtered or deemed unsuitable by the API)."
        if "invalid Part" in error_str or "content must be provided" in error_str or "Invalid JSON payload" in error_str:
             logger.error(f"Gemini API Error: Invalid input structure or payload. Raw Error: {error_str}")
             return None, "Error: There was an issue with the data sent to the Gemini API. Please try again or check attachments."
        if "candidate" in error_str.lower() and ("index out of range" in error_str or "NoneType" in error_str or "has no attribute 'text'" in error_str):
             logger.warning(f"Gemini API Error: Could not extract text from candidate, possibly due to filtering. Raw Error: {error_str}")
             return None, "Error: Could not generate response from Gemini (Your request might have been filtered or deemed unsuitable by the API)."
        if "does not support image input" in error_str.lower() or ("User input is not supported" in error_str and "image" in error_str.lower()):
             logger.error(f"Model {effective_model_name} does not support image input, but images were provided.")
             return None, f"Error: The current model (`{effective_model_name}`) does not support image input. Please remove the image or change the model using `setmodel`."
        if "grpc" in error_str.lower() or "Deadline Exceeded" in error_str or "503" in error_str or "unavailable" in error_str.lower(): # Network/service issues
             return None, f"Error: Network connection issue or timeout with Gemini. Please try again. (Details: {error_str[:50]})"
        
        # Fallback generic error
        return None, f"An error occurred calling the Gemini API ({type(e).__name__}). Please try again later. (Details: {error_str[:100]})"


async def create_streaming_footer(processing_time: float, word_count: int, effective_model_name: str) -> str:
    """Creates the footer for streaming messages."""
    footer_base = (f"> {EMOJI_CLOCK} {processing_time:.2f} seconds\n"
                   f"> {EMOJI_NEKO_EARS} Model: {effective_model_name}\n"
                   f"> {EMOJI_KEYBOARD_CAT} {word_count} words\n")
    return f"## {EMOJI_BARD_THINK} Processing...\n{footer_base}"

async def create_final_footer(processing_time: float, word_count: int, effective_model_name: str) -> str:
    """Creates the footer for the final message."""
    footer_base = (f"> {EMOJI_CLOCK} {processing_time:.2f} seconds\n"
                   f"> {EMOJI_NEKO_EARS} Model: {effective_model_name}\n"
                   f"> {EMOJI_KEYBOARD_CAT} {word_count} words\n")
    return f"{footer_base}" # No "Processing..." header

# --- Text Splitting Logic ---
def find_best_split_point(text: str, max_len: int) -> int:
    """Finds the best point to split text, avoiding mid-code-block splits."""
    if len(text) <= max_len: return len(text)

    code_block_spans = []
    # Regex to find ``` optionally followed by lang, newline, content, then ```
    for match in re.finditer(r"```(?:[\w\+\#-]+)?\s*?\n.*?```", text, re.DOTALL | re.MULTILINE):
        code_block_spans.append(match.span())

    def is_inside_block(index_to_check):
        # Check if the character *at* index_to_check is inside a code block,
        # excluding the ``` markers themselves.
        for start, end in code_block_spans:
            if start + 3 <= index_to_check < end - 3: # strictly inside
                return True
        return False

    # Prefer splitting at these delimiters, in order of preference (roughly)
    delimiters = ["\n\n", "\n", ". ", "? ", "! ", "—", "–", ") ", "] ", "} ", " "]
    best_split_point = 0
    
    # Search for delimiters within a reasonable window before max_len
    # Reduced search window slightly to be more targeted
    search_window_start = max(0, max_len - 200) 

    for delim in delimiters:
        try:
            # Find the last occurrence of the delimiter before or at max_len
            split_candidate = text.rindex(delim, search_window_start, max_len)
            # Position *after* the delimiter
            pos_after_delimiter = split_candidate + len(delim) 
            
            if not is_inside_block(pos_after_delimiter -1): # Check char before split
                if pos_after_delimiter > best_split_point:
                    best_split_point = pos_after_delimiter
                # If a major delimiter (like newline or sentence end) is found close to max_len, take it
                if delim in ["\n\n", "\n", ". ", "? ", "! "] and (max_len - best_split_point < 100):
                    return best_split_point
        except ValueError:
            continue # Delimiter not found in the range

    if best_split_point > 0: return best_split_point
    
    # If no delimiter found, or all are inside code blocks, try to split at max_len
    # if it's not inside a code block.
    if not is_inside_block(max_len -1): return max_len

    # Fallback: find last space before max_len that's not in a code block
    current_pos = max_len -1
    while current_pos > 0:
        if text[current_pos] == ' ' and not is_inside_block(current_pos):
            return current_pos + 1
        current_pos -=1
    
    logger.warning(f"Could not find a good split point for text of len {len(text)} with max_len {max_len}. Force splitting at max_len.")
    return max_len # Absolute fallback

def is_code_block_open(text: str) -> bool:
    """Checks if the text ends with an open code block."""
    last_fence_pos = text.rfind("```")
    if last_fence_pos == -1: return False # No code fences
    # Count occurrences of ``` before the last one. If even, the last one is opening.
    num_fences_before = text.count("```", 0, last_fence_pos)
    return num_fences_before % 2 == 0

def get_code_block_language(text: str) -> str:
    """Extracts the language from the last opened code block's header."""
    language = ""
    # Find the position of the last opening ```
    last_opening_fence_pos = -1
    search_pos = len(text)
    while True:
        temp_pos = text.rfind("```", 0, search_pos)
        if temp_pos == -1: break # No more fences
        
        count_before = text.count("```", 0, temp_pos)
        if count_before % 2 == 0: # This is an opening fence
            last_opening_fence_pos = temp_pos
            break
        search_pos = temp_pos # Continue searching before this closing fence

    if last_opening_fence_pos != -1:
        # Find the end of the language line (newline)
        header_end_pos = text.find("\n", last_opening_fence_pos + 3) # Start search after ```
        if header_end_pos != -1:
            potential_header = text[last_opening_fence_pos + 3 : header_end_pos].strip()
            # Basic validation for a language string (alphanumeric, -, #, +)
            if potential_header and re.match(r"^[\w\+\#-]+$", potential_header):
                language = potential_header
    return language

# --- Streaming Response Handler ---
async def _handle_streaming_response(
    target_message: discord.Message | discord.InteractionMessage, # User's message or Interaction's original response
    initial_bot_message: Optional[discord.Message], # Bot's "thinking" message or Interaction's original response
    response_stream: GenerateContentResponse,
    start_time: float,
    effective_model_name: str,
    interaction: Optional[discord.Interaction] = None # Pass the interaction object if applicable
) -> Optional[str]:
    full_response = ""
    pending_text = ""
    current_message_content = "" # Content of the current bot message being edited/built
    last_edit_time = time.time()
    
    # edit_this_message is the discord.Message object we are currently editing or will edit next
    edit_this_message = initial_bot_message 
    bot_messages: list[discord.Message] = [initial_bot_message] if initial_bot_message else []
    
    first_chunk_processed = False
    channel = interaction.channel if interaction else target_message.channel # Get channel from interaction or target

    # If initial_bot_message (placeholder) couldn't be sent/retrieved
    if not edit_this_message and channel:
        logger.warning(f"_handle_streaming_response called with initial_bot_message=None for {'interaction' if interaction else 'message context'}. Attempting to send new placeholder.")
        placeholder_content = f"## {EMOJI_LUNA_THINKING} Processing..." # Initial placeholder text
        try:
            if interaction:
                logger.info("Sending initial followup message for interaction as placeholder.")
                # This followup becomes the first message of the bot's response stream
                edit_this_message = await interaction.followup.send(placeholder_content, wait=True) 
            elif isinstance(target_message, discord.Message): # For prefix commands or on_message
                logger.info("Sending initial message as placeholder for prefix command/message.")
                edit_this_message = await channel.send(placeholder_content, reference=target_message, mention_author=False)
            else: # Should not happen if called correctly
                logger.error("Cannot send initial placeholder: target_message is not a discord.Message and no interaction provided.")
                # Attempt to notify user if possible, then return None as we can't proceed
                if interaction: await interaction.followup.send("Error: Could not initialize response message.", ephemeral=True)
                return None

            if edit_this_message:
                bot_messages = [edit_this_message] # Reset bot_messages to only contain the new placeholder
            else: # If sending placeholder failed
                logger.error("Failed to send the new placeholder message.")
                if interaction: await interaction.followup.send("Error: Failed to send initial response message.", ephemeral=True)
                return None

        except discord.HTTPException as http_e:
            logger.error(f"HTTP Error sending initial placeholder message: {http_e.status} - {http_e.text}", exc_info=True)
            if interaction: await interaction.followup.send(f"Error: Could not send initial response message (HTTP {http_e.status}).", ephemeral=True)
            return None
        except Exception as e_placeholder: # Catch any other errors during placeholder send
            logger.error(f"Unexpected error sending initial placeholder message: {e_placeholder}", exc_info=True)
            if interaction: await interaction.followup.send("Error: An unexpected issue occurred initializing the response.", ephemeral=True)
            return None

    # Determine the reference for new messages (only for non-interaction messages)
    message_reference = target_message if isinstance(target_message, discord.Message) and not interaction else None
    
    original_interaction_response_id: Optional[int] = None
    if interaction:
        try:
            # Get the ID of the initial deferred response message to correctly use edit_original_response
            original_interaction_response_msg = await interaction.original_response()
            original_interaction_response_id = original_interaction_response_msg.id
        except (discord.NotFound, discord.HTTPException) as e:
            logger.warning(f"Could not get original interaction response for ID: {e}")
        except Exception as e_orig_resp: # Catch any other error
            logger.error(f"Unexpected error getting original interaction response: {e_orig_resp}", exc_info=True)


    try: # Outer try for the whole streaming loop
        async for chunk in response_stream:
            chunk_text_current = ""
            try: # Extract text from chunk safely
                if chunk.parts: 
                    chunk_text_current = ''.join(part.text for part in chunk.parts if hasattr(part, 'text'))
                elif hasattr(chunk, 'text') and chunk.text: 
                    chunk_text_current = chunk.text
                # Fallback for some response structures
                elif chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts: 
                    chunk_text_current = ''.join(part.text for part in chunk.candidates[0].content.parts if hasattr(part, 'text'))
            except ValueError as ve: # E.g., if .text is accessed on a part blocked by safety
                 if "text is not available" in str(ve).lower(): 
                     logger.warning("Stream chunk contained non-text part (likely safety filtered), skipping text extraction for this part.")
                 else:
                     logger.warning(f"ValueError extracting text from stream chunk: {ve}", exc_info=False)
                 continue # Skip this problematic part of the chunk
            except AttributeError: # If expected attributes are missing
                logger.warning("AttributeError extracting text from stream chunk, structure unexpected.", exc_info=False)
                continue
            except Exception as e_chunk_text: # Catch-all for other text extraction issues
                logger.warning(f"Unexpected error extracting text from stream chunk: {e_chunk_text}", exc_info=False)
                continue
            
            if not chunk_text_current: continue # Skip empty chunks

            full_response += chunk_text_current
            pending_text += chunk_text_current
            
            needs_edit = False
            force_edit_now = False # Force edit for the very first chunk
            needs_new_message = False
            
            current_time = time.time()
            current_time_for_footer = current_time - start_time
            current_word_count = len(full_response.split())
            
            streaming_footer = await create_streaming_footer(current_time_for_footer, current_word_count, effective_model_name)
            footer_len_estimate = len("\n" + streaming_footer) + 20 # Estimate for footer and some buffer

            combined_text_for_current_msg = current_message_content + pending_text
            projected_len_for_current_msg = len(combined_text_for_current_msg) + footer_len_estimate

            if projected_len_for_current_msg > MESSAGE_LENGTH_LIMIT:
                needs_new_message = True
            elif not first_chunk_processed or (current_time - last_edit_time > EDIT_INTERVAL):
                if pending_text: # Only edit if there's new text
                    needs_edit = True
                if not first_chunk_processed:
                    force_edit_now = True # Ensure the first piece of content is shown quickly

            # --- Handle Sending a New Message due to Length ---
            if needs_new_message:
                # Determine split point for the combined text intended for the current message
                split_point_in_combined = find_best_split_point(combined_text_for_current_msg, MESSAGE_LENGTH_LIMIT - footer_len_estimate - 10) # -10 for safety margin
                if split_point_in_combined <= 0 : # Should not happen with find_best_split_point's fallbacks
                    logger.warning(f"find_best_split_point returned {split_point_in_combined}. Forcing split near limit."); 
                    split_point_in_combined = min(len(combined_text_for_current_msg), MESSAGE_LENGTH_LIMIT - footer_len_estimate - 30)
                
                # Ensure split point is within bounds
                split_point_in_combined = max(0, min(split_point_in_combined, len(combined_text_for_current_msg)))

                text_for_current_message_final_edit: str
                text_for_next_message_initial_content: str

                if split_point_in_combined <= len(current_message_content): 
                    # This implies pending_text is too large to append even partially, or split is at current_message_content end
                    text_for_current_message_final_edit = current_message_content
                    text_for_next_message_initial_content = pending_text 
                else:
                    # Normal split within combined_text_for_current_msg
                    text_for_current_message_final_edit = combined_text_for_current_msg[:split_point_in_combined].rstrip()
                    text_for_next_message_initial_content = combined_text_for_current_msg[split_point_in_combined:].lstrip()

                display_text_for_current_final_edit = text_for_current_message_final_edit
                current_message_ended_in_open_code_block = is_code_block_open(text_for_current_message_final_edit)
                language_for_next_message_code_block = ""

                if current_message_ended_in_open_code_block:
                    if not display_text_for_current_final_edit.endswith("\n```"): # Ensure it ends cleanly with closing fence
                        display_text_for_current_final_edit = display_text_for_current_final_edit.rstrip() + "\n```"
                    language_for_next_message_code_block = get_code_block_language(text_for_current_message_final_edit)
                    text_for_next_message_initial_content = f"```{language_for_next_message_code_block}\n{text_for_next_message_initial_content}"
                
                # Finalize the current message (before starting a new one)
                try:
                    if edit_this_message:
                        final_edit_content_current_msg = display_text_for_current_final_edit[:MESSAGE_LENGTH_LIMIT] # Ensure it fits
                        if edit_this_message.content.strip() != final_edit_content_current_msg.strip(): # Avoid redundant edits
                            if interaction and original_interaction_response_id and edit_this_message.id == original_interaction_response_id:
                                await interaction.edit_original_response(content=final_edit_content_current_msg)
                            else:
                                await edit_this_message.edit(content=final_edit_content_current_msg)
                    else:
                        logger.warning("edit_this_message became None before final edit during split.")
                except discord.NotFound:
                    logger.warning(f"Message {(edit_this_message.id if edit_this_message else 'N/A')} not found for final edit before new message. Current message lost.");
                    edit_this_message = None # Mark as lost
                except discord.HTTPException as e_final_edit:
                    logger.error(f"HTTP Error finalizing message {edit_this_message.id if edit_this_message else 'N/A'} before split: {e_final_edit.status} - {e_final_edit.text}")
                except Exception as e_final_edit_unexpected:
                    logger.error(f"Unexpected error finalizing message {edit_this_message.id if edit_this_message else 'N/A'} before split: {e_final_edit_unexpected}", exc_info=True)

                # Prepare content for the new message
                content_for_new_message_display_text = text_for_next_message_initial_content
                # If the *start* of the new message is an open code block, ensure it's properly closed for display
                if is_code_block_open(content_for_new_message_display_text):
                    if not content_for_new_message_display_text.endswith("\n```"):
                         content_for_new_message_display_text = content_for_new_message_display_text.rstrip() + "\n```"
                
                streaming_footer_for_new_message = await create_streaming_footer(current_time_for_footer, current_word_count, effective_model_name)
                content_for_new_message_with_footer = content_for_new_message_display_text + "\n" + streaming_footer_for_new_message

                if len(content_for_new_message_with_footer) > MESSAGE_LENGTH_LIMIT:
                    logger.warning(f"New message content still exceeds limit ({len(content_for_new_message_with_footer)}) even after split. Truncating initial part of new message.");
                    cutoff = MESSAGE_LENGTH_LIMIT - len(streaming_footer_for_new_message) - 50 # Generous buffer for safety
                    text_for_next_message_truncated = text_for_next_message_initial_content[:cutoff] + "... (truncated)"
                    
                    display_text_next_truncated = text_for_next_message_truncated
                    if is_code_block_open(display_text_next_truncated): # Re-check if truncation affects code block
                        if not display_text_next_truncated.endswith("\n```"):
                            display_text_next_truncated = display_text_next_truncated.rstrip() + "\n```"
                    
                    content_for_new_message_with_footer = display_text_next_truncated + "\n" + streaming_footer_for_new_message
                    text_for_next_message_initial_content = text_for_next_message_truncated # Update the actual content part

                if not text_for_next_message_initial_content.strip(): # If split resulted in empty content for new message
                    logger.warning("Split resulted in empty text for the new message. Skipping new message send.");
                    pending_text = "" # Clear pending as it was meant for the new message
                    current_message_content = text_for_current_message_final_edit # Update current message content
                    needs_edit = False # No immediate edit needed as we just finalized current
                    continue # Continue to next chunk

                # --- Send the new message part ---
                new_bot_message: Optional[discord.Message] = None
                try:
                    if interaction: 
                        new_bot_message = await interaction.followup.send(content=content_for_new_message_with_footer, wait=True)
                    elif channel: # For prefix commands or on_message
                        new_bot_message = await channel.send(content=content_for_new_message_with_footer, reference=message_reference, mention_author=False)
                    
                    if new_bot_message:
                        bot_messages.append(new_bot_message)
                        edit_this_message = new_bot_message # Future edits target this new message
                        current_message_content = text_for_next_message_initial_content # Base content for the new message
                        pending_text = "" # Reset pending_text as it's now part of current_message_content
                        last_edit_time = time.time()
                        if not first_chunk_processed: first_chunk_processed = True
                    else:
                        logger.error("Failed to send new message part (send call returned None or channel was None).")
                        # Attempt to notify, then break as stream is broken
                        if interaction: await interaction.followup.send("Error: Could not send the next part of the message.", ephemeral=True)
                        elif channel: await channel.send("Error: Could not send the next part of the message.", reference=message_reference, mention_author=False)
                        break 

                except discord.HTTPException as e_new_send_http:
                    logger.error(f"HTTP Error sending new message part: {e_new_send_http.status} - {e_new_send_http.text}")
                    try: # Attempt to inform user
                        err_msg_user = "Error: A network issue occurred while sending the next message part."
                        if interaction: await interaction.followup.send(err_msg_user, ephemeral=True)
                        elif channel: await channel.send(err_msg_user, reference=message_reference, mention_author=False)
                    except Exception: pass
                    pending_text = "" # Discard text that couldn't be sent
                    break # Stop streaming for this response
                except Exception as e_new_send_unexpected:
                     logger.error(f"Unexpected error sending new message part: {e_new_send_unexpected}", exc_info=True)
                     pending_text = ""
                     break # Stop streaming
                # --- End of sending new message ---
                needs_edit = False # Reset as we just started a new message

            # --- Handle Editing Existing Message (Not a new message) ---
            elif needs_edit and edit_this_message:
                content_to_edit_current_msg = current_message_content + pending_text
                display_content_for_edit = content_to_edit_current_msg
                
                if is_code_block_open(display_content_for_edit): # Ensure open code blocks are terminated for display
                    if not display_content_for_edit.endswith("\n```"):
                        display_content_for_edit = display_content_for_edit.rstrip() + "\n```"
                
                streaming_footer_for_edit = await create_streaming_footer(current_time_for_footer, current_word_count, effective_model_name)
                content_to_edit_with_footer = display_content_for_edit + "\n" + streaming_footer_for_edit

                if len(content_to_edit_with_footer) <= MESSAGE_LENGTH_LIMIT:
                    should_actually_edit_now = False
                    try: # Check if content has changed to avoid redundant edits
                        if force_edit_now or edit_this_message.content.strip() != content_to_edit_with_footer.strip():
                            should_actually_edit_now = True
                    except AttributeError: # edit_this_message might be None if initial send failed badly
                        logger.warning("Attempted to check content for edit on a None message. Skipping edit check.")
                        should_actually_edit_now = False 
                    
                    if should_actually_edit_now:
                        try:
                            if interaction and original_interaction_response_id and edit_this_message.id == original_interaction_response_id:
                                await interaction.edit_original_response(content=content_to_edit_with_footer)
                            else:
                                await edit_this_message.edit(content=content_to_edit_with_footer)
                            
                            last_edit_time = current_time
                            current_message_content = content_to_edit_current_msg # Update base content
                            pending_text = "" # Clear pending as it's now part of current_message_content
                            if not first_chunk_processed: first_chunk_processed = True
                        except discord.NotFound:
                            logger.warning(f"Message {edit_this_message.id} not found for edit. Stopping stream for this message.");
                            break # Message is gone, can't continue editing it
                        except discord.HTTPException as e_edit_http:
                            if e_edit_http.status == 429: # Rate limited
                                retry_after = getattr(e_edit_http, 'retry_after', EDIT_INTERVAL * 2) # Use discord.py's retry_after if available
                                logger.warning(f"Rate limited on edit for message {edit_this_message.id}. Waiting {retry_after:.2f}s. Pending text will accumulate.")
                                await asyncio.sleep(retry_after)
                                # Do not reset last_edit_time here, let the loop retry edit on next eligible cycle
                            else:
                                logger.warning(f"HTTP Error editing message {edit_this_message.id}: {e_edit_http.status} - {e_edit_http.text}. Pending text will accumulate.")
                                last_edit_time = current_time # Update time to prevent rapid retries if error persists
                        except Exception as e_edit_unexpected:
                            logger.error(f"Unexpected error editing message {edit_this_message.id}: {e_edit_unexpected}", exc_info=True)
                            last_edit_time = current_time # Update time to prevent rapid retries
                else: # Content too long for edit, defer to next cycle (which should trigger a new message)
                    logger.warning(f"Content too long for edit ({len(content_to_edit_with_footer)}), deferring. Next cycle should split.")
                    last_edit_time = current_time # Mark that an attempt was made

            # Safety check: if edit_this_message somehow got lost from our tracking
            if edit_this_message and not any(m.id == edit_this_message.id for m in bot_messages if m):
                logger.warning(f"Current edit target message {edit_this_message.id} is no longer in tracked bot_messages. Stopping stream.");
                break
        # --- End of async for chunk in response_stream loop ---

        # --- Stream Finished ---
        logger.info(f"Stream finished for source message/interaction ID {target_message.id if isinstance(target_message, (discord.Message, discord.InteractionMessage)) else interaction.id if interaction else 'UnknownSource'}. Total response length: {len(full_response)}")
        total_processing_time = time.time() - start_time
        total_word_count = len(full_response.split())
        final_footer_text = await create_final_footer(total_processing_time, total_word_count, effective_model_name)

        # --- Final Update of the Last Message ---
        if not edit_this_message or not any(m.id == edit_this_message.id for m in bot_messages if m): # Check if last message is still valid
             logger.error("Last bot message for stream was lost or became invalid before final update.")
             final_content_orphan = (current_message_content + pending_text + "\n" + final_footer_text)[:MESSAGE_LENGTH_LIMIT]
             if final_content_orphan.strip() and channel:
                 try:
                      logger.info("Attempting to send final orphaned part as a new message.");
                      if interaction: await interaction.followup.send(content=final_content_orphan) # No ephemeral for main content
                      elif channel: await channel.send(content=final_content_orphan, reference=message_reference, mention_author=False)
                 except Exception as e_send_orphan: 
                      logger.error(f"Failed sending final orphaned part as new message: {e_send_orphan}", exc_info=True)
             return full_response # Return the full response text even if sending failed

        # Combine any remaining pending text with the content of the last message
        final_content_for_last_msg = current_message_content + pending_text
        final_content_for_last_msg_with_footer = final_content_for_last_msg + "\n" + final_footer_text

        if len(final_content_for_last_msg_with_footer) <= MESSAGE_LENGTH_LIMIT: # If it fits, edit the last message
            try:
                if edit_this_message.content.strip() != final_content_for_last_msg_with_footer.strip():
                    if interaction and original_interaction_response_id and edit_this_message.id == original_interaction_response_id:
                        await interaction.edit_original_response(content=final_content_for_last_msg_with_footer)
                    else:
                        await edit_this_message.edit(content=final_content_for_last_msg_with_footer)
            except discord.NotFound:
                 logger.warning(f"Final message {edit_this_message.id} not found for final edit.")
                 if channel: # Try sending this final content as a new message
                     try: 
                          final_content_safe_send = final_content_for_last_msg_with_footer[:MESSAGE_LENGTH_LIMIT]
                          logger.info("Sending final content as new message due to NotFound on edit.")
                          if interaction: await interaction.followup.send(content=final_content_safe_send)
                          elif channel: await channel.send(content=final_content_safe_send, reference=message_reference, mention_author=False)
                     except Exception as e_send_final_new: 
                          logger.error(f"Failed sending final part as new message after NotFound: {e_send_final_new}", exc_info=True)
            except discord.HTTPException as e_final_edit_http:
                logger.error(f"HTTP Error on final edit of message {edit_this_message.id}: {e_final_edit_http.status} - {e_final_edit_http.text}")
            except Exception as e_final_edit_unexpected:
                logger.error(f"Unexpected error on final edit of message {edit_this_message.id}: {e_final_edit_unexpected}", exc_info=True)
        else: # Final content is too long, requires one last split
            logger.info(f"Final content too long ({len(final_content_for_last_msg_with_footer)}). Performing final split.")
            
            text_part_for_final_split = final_content_for_last_msg # Text without footer for splitting
            # Split for the existing message, leave room for potential code block closers
            split_point_final_edit = find_best_split_point(text_part_for_final_split, MESSAGE_LENGTH_LIMIT - 20) 
            if split_point_final_edit <= 0: # Fallback if split point is problematic
                split_point_final_edit = max(0, MESSAGE_LENGTH_LIMIT - len(final_footer_text) - 30) # Ensure room for footer on next part
                logger.warning(f"Forcing final split point at {split_point_final_edit}")

            part1_for_final_edit = text_part_for_final_split[:split_point_final_edit].rstrip()
            part2_for_new_message_final = text_part_for_final_split[split_point_final_edit:].lstrip()
            
            display_part1_final_edit = part1_for_final_edit
            part1_ended_open_code = is_code_block_open(part1_for_final_edit)
            language_for_part2_code = ""

            if part1_ended_open_code:
                if not display_part1_final_edit.endswith("\n```"):
                    display_part1_final_edit = display_part1_final_edit.rstrip() + "\n```"
                language_for_part2_code = get_code_block_language(part1_for_final_edit)
                part2_for_new_message_final = f"```{language_for_part2_code}\n{part2_for_new_message_final}"
            
            # Edit the last current message with Part 1
            try:
                 final_edit_part1_safe = display_part1_final_edit[:MESSAGE_LENGTH_LIMIT]
                 if edit_this_message.content.strip() != final_edit_part1_safe.strip():
                     if interaction and original_interaction_response_id and edit_this_message.id == original_interaction_response_id:
                         await interaction.edit_original_response(content=final_edit_part1_safe)
                     else:
                         await edit_this_message.edit(content=final_edit_part1_safe)
            except discord.NotFound:
                logger.warning(f"Final message {edit_this_message.id} not found for split edit (part 1). Part 2 may be lost or sent standalone.")
                # Part 2 will attempt to send as a new message below.
            except discord.HTTPException as e_final_split_edit_http:
                logger.error(f"HTTP Error editing final message part 1 (split): {e_final_split_edit_http.status} - {e_final_split_edit_http.text}")
            except Exception as e_final_split_edit_unexpected:
                logger.error(f"Unexpected error editing final message part 1 (split): {e_final_split_edit_unexpected}", exc_info=True)
            
            # Send Part 2 + Final Footer as a new message
            part2_final_with_footer = part2_for_new_message_final + "\n" + final_footer_text
            try:
                 content_for_final_new_message = part2_final_with_footer[:MESSAGE_LENGTH_LIMIT]
                 if len(content_for_final_new_message) < len(part2_final_with_footer): # Log if truncated
                     logger.warning(f"Truncating final new message (part 2 of split) as it exceeded limit.")
                 
                 if content_for_final_new_message.strip() and channel: # Ensure there's content to send
                    if interaction: await interaction.followup.send(content=content_for_final_new_message)
                    elif channel: await channel.send(content=content_for_final_new_message, reference=message_reference, mention_author=False) # Reference original user message if possible
            except discord.HTTPException as e_final_send_part2_http:
                 logger.error(f"HTTP Error sending final message part 2 (split): {e_final_send_part2_http.status} - {e_final_send_part2_http.text}")
                 try: # Inform user about the issue with the last part
                      err_msg_user_final = "Error: A network issue occurred sending the very last part of the message."
                      if interaction: await interaction.followup.send(err_msg_user_final, ephemeral=True)
                      elif channel: await channel.send(err_msg_user_final, reference=message_reference, mention_author=False)
                 except Exception: pass
            except Exception as e_final_send_part2_unexpected:
                 logger.error(f"Unexpected error sending final message part 2 (split): {e_final_send_part2_unexpected}", exc_info=True)

        return full_response

    # --- Exception Handling for the entire streaming block ---
    except Exception as e_streaming_main: # Catch-all for unexpected errors during the streaming loop itself
        logger.error(f"Generic error during streaming for {target_message.id if isinstance(target_message, (discord.Message, discord.InteractionMessage)) else interaction.id if interaction else 'UnknownSource'}: {e_streaming_main}", exc_info=True)
        error_message_content = f"Oops! An unexpected error occurred while I was replying: `{str(e_streaming_main)[:150]}`" # Keep error brief for user
        
        try: # Attempt to notify user by editing the last known message or sending a new one
             last_bot_msg_for_error = bot_messages[-1] if bot_messages and bot_messages[-1] else None
             
             # Prefer editing the message that was being worked on, or the initial placeholder if nothing else
             target_edit_msg_for_error = edit_this_message if edit_this_message else last_bot_msg_for_error
             if not target_edit_msg_for_error and not first_chunk_processed: # If very early error
                target_edit_msg_for_error = initial_bot_message


             if target_edit_msg_for_error and channel: # If we have a message to edit
                 # Clean up any "Processing..." footer before adding the error
                 content_of_target_error_msg = ""
                 try: content_of_target_error_msg = target_edit_msg_for_error.content
                 except Exception: pass # Ignore if content access fails

                 cleaned_content_for_error = re.sub(r"\n## "+re.escape(EMOJI_BARD_THINK)+r".*", "", content_of_target_error_msg, flags=re.DOTALL).rstrip()
                 error_edit_content = f"{cleaned_content_for_error}\n\n{error_message_content}"[:MESSAGE_LENGTH_LIMIT]
                 
                 if interaction and original_interaction_response_id and target_edit_msg_for_error.id == original_interaction_response_id:
                     await interaction.edit_original_response(content=error_edit_content)
                 else:
                     await target_edit_msg_for_error.edit(content=error_edit_content)
             elif channel: # Fallback to sending a new message if no suitable message to edit
                 if interaction: await interaction.followup.send(content=error_message_content, ephemeral=True) # Ephemeral if it's a new error message for interaction
                 elif channel: await channel.send(content=error_message_content, reference=message_reference, mention_author=False)
        except Exception as final_error_report_ex:
            logger.error(f"Failed to send/edit the generic error message to the user: {final_error_report_ex}", exc_info=True)
        
        return None # Indicate failure


# --- Attachment Processing ---
async def _process_attachments(attachments: list[discord.Attachment]) -> tuple[list[str], list[tuple[str, bytes]], list[str], bool]:
    """Processes message attachments, separating text files and images."""
    files_for_prompt: List[str] = []
    images_for_prompt: List[Tuple[str, bytes]] = []
    attachment_errors: List[str] = []
    unsupported_non_image_found = False
    processed_attachments_count = 0

    if not attachments: 
        return files_for_prompt, images_for_prompt, attachment_errors, unsupported_non_image_found

    logger.info(f"Processing {len(attachments)} attachment(s)...")
    for attachment in attachments:
        if processed_attachments_count >= MAX_ATTACHMENTS_PER_MESSAGE:
            if processed_attachments_count == MAX_ATTACHMENTS_PER_MESSAGE: # Log only once
                err_msg = f"Skipped further attachments: Maximum of {MAX_ATTACHMENTS_PER_MESSAGE} attachments processed."
                attachment_errors.append(err_msg)
                logger.warning(err_msg)
            processed_attachments_count += 1
            continue # Skip if max attachments already processed
        
        processed_attachments_count += 1
        file_size_mb = attachment.size / (1024 * 1024)
        logger.debug(f"Processing attachment: {attachment.filename} (Type: {attachment.content_type}, Size: {file_size_mb:.2f} MB)")

        try:
            is_image = attachment.content_type and attachment.content_type in SUPPORTED_IMAGE_TYPES
            is_text_file = (attachment.content_type and attachment.content_type.startswith("text/")) or \
                           any(attachment.filename.lower().endswith(ext) for ext in SUPPORTED_TEXT_EXTENSIONS) or \
                           (attachment.content_type in ["application/json", "application/x-yaml", "application/yaml", "application/xml", "application/javascript"])

            if is_image:
                if file_size_mb > MAX_IMAGE_SIZE_MB:
                    logger.warning(f"Image attachment too large: {attachment.filename} ({file_size_mb:.2f}MB > {MAX_IMAGE_SIZE_MB}MB). Skipping.")
                    attachment_errors.append(f"Image `{attachment.filename}` is too large (max {MAX_IMAGE_SIZE_MB}MB).")
                    continue
                try:
                    image_bytes = await attachment.read()
                    # Double check size after reading, though attachment.size should be accurate
                    if len(image_bytes) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
                        logger.warning(f"Image {attachment.filename} exceeded size limit after read. Skipping.")
                        attachment_errors.append(f"Image `{attachment.filename}` exceeded size limit after read (max {MAX_IMAGE_SIZE_MB}MB).")
                        continue
                    images_for_prompt.append((attachment.content_type, image_bytes))
                    logger.debug(f"Added image attachment: {attachment.filename}")
                except Exception as e_read_img:
                    logger.error(f"Error reading image attachment {attachment.filename}: {e_read_img}", exc_info=True)
                    attachment_errors.append(f"Error reading image `{attachment.filename}`.")
            
            elif is_text_file:
                if file_size_mb > MAX_TEXT_FILE_SIZE_MB:
                    logger.warning(f"Text file attachment too large: {attachment.filename} ({file_size_mb:.2f}MB > {MAX_TEXT_FILE_SIZE_MB}MB). Skipping.")
                    attachment_errors.append(f"Text file `{attachment.filename}` is too large (max {MAX_TEXT_FILE_SIZE_MB}MB).")
                    continue
                
                file_extension = os.path.splitext(attachment.filename)[1].lower().lstrip('.')
                language_hint = LANGUAGE_MAPPINGS.get(file_extension, "text") # Default to "text" if no specific mapping
                logger.debug(f"Determined language hint for {attachment.filename} (ext: '{file_extension}'): '{language_hint}'.")
                
                try:
                    file_bytes = await attachment.read()
                    if len(file_bytes) / (1024 * 1024) > MAX_TEXT_FILE_SIZE_MB: # Double check size
                        logger.warning(f"Text file {attachment.filename} exceeded size limit after read. Skipping.")
                        attachment_errors.append(f"Text file `{attachment.filename}` exceeded size limit after read (max {MAX_TEXT_FILE_SIZE_MB}MB).")
                        continue

                    decoded_content = ""
                    try: 
                        decoded_content = file_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.warning(f"UTF-8 decoding failed for {attachment.filename}. Trying fallbacks (latin-1, cp1252)...")
                        try: 
                            decoded_content = file_bytes.decode('latin-1')
                        except UnicodeDecodeError:
                             try: 
                                 decoded_content = file_bytes.decode('cp1252')
                             except Exception as e_decode_fallback:
                                 logger.error(f"All decoding fallbacks failed for {attachment.filename}: {e_decode_fallback}", exc_info=True)
                                 attachment_errors.append(f"Could not decode text file `{attachment.filename}`.")
                                 continue # Skip this file
                    
                    files_for_prompt.append(f"--- File: `{attachment.filename}` ---\n```{language_hint}\n{decoded_content}\n```\n--- End File ---")
                    logger.debug(f"Added text file attachment: {attachment.filename}")
                except Exception as e_read_text:
                    logger.error(f"Error reading/decoding text file {attachment.filename}: {e_read_text}", exc_info=True)
                    attachment_errors.append(f"Error reading text file `{attachment.filename}`.")
            
            else: # Unsupported attachment type
                logger.info(f"Skipping unsupported attachment type: {attachment.filename} (Content-Type: {attachment.content_type or 'N/A'})")
                if not is_image: # Track if it was a non-image unsupported type
                    unsupported_non_image_found = True
        
        except Exception as e_process_attach: # Catch-all for errors during the processing of a single attachment
            logger.error(f"General error processing attachment {attachment.filename}: {e_process_attach}", exc_info=True)
            attachment_errors.append(f"An error occurred while processing file `{attachment.filename}`.")

    if processed_attachments_count > MAX_ATTACHMENTS_PER_MESSAGE:
        logger.warning(f"Total attachments processed: {MAX_ATTACHMENTS_PER_MESSAGE}. Skipped {processed_attachments_count - MAX_ATTACHMENTS_PER_MESSAGE} additional attachments due to limit.")
    
    return files_for_prompt, images_for_prompt, attachment_errors, unsupported_non_image_found


# --- Function to Format Context History ---
def format_context_history(channel_id: int, max_tokens: int) -> str:
    """Formats channel context history for the prompt, respecting token limits and time."""
    history_deque = CONTEXT_MEMORY.get(channel_id)
    if not history_deque: 
        return ""

    formatted_lines: List[str] = []
    current_token_count = 0
    # Time limit: messages older than this are not included
    oldest_permissible_timestamp = time.time() - CONTEXT_TIME_LIMIT_SECONDS 

    # Iterate from newest to oldest (deque is appended to right, so iterate reversed)
    for entry in reversed(history_deque):
        timestamp, user_id, user_name, role, content = entry
        
        if timestamp < oldest_permissible_timestamp:
            logger.debug(f"Context entry from {datetime.fromtimestamp(timestamp)} for Ch {channel_id} exceeds time limit. Truncating history here.")
            formatted_lines.append("... (older messages truncated due to time limit)")
            break # Stop processing older messages

        # Rough token estimate: words + some overhead for formatting/role
        # A more accurate tokenizer would be better but adds complexity.
        entry_tokens = len(content.split()) + 10 # Estimate
        
        if current_token_count + entry_tokens > max_tokens:
            logger.debug(f"Context history for Ch {channel_id} reached token limit ({current_token_count}+{entry_tokens} > {max_tokens}). Truncating history here.")
            formatted_lines.append("... (further messages truncated due to token limit)")
            break # Stop processing if token limit exceeded

        # Format: "Role (UserName): Content" or "Role: Content" if bot
        formatted_line = f"{role} ({user_name}): {content}" if role == "User" else f"{role}: {content}"
        formatted_lines.append(formatted_line)
        current_token_count += entry_tokens
    
    if not formatted_lines: 
        return ""
    
    # Return history in chronological order (oldest relevant to newest relevant)
    return "\n".join(reversed(formatted_lines))


# --- Message Processing Logic (for non-command messages) ---
async def process_message(message: discord.Message):
    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id

    # Permission checks for guild channels
    if isinstance(message.channel, discord.TextChannel):
        if not message.guild: # Should not happen if TextChannel, but defensive
            logger.warning(f"TextChannel {channel_id} has no guild attribute. Skipping message {message.id}.")
            return 
        # Check bot's permissions in the channel
        permissions = message.channel.permissions_for(message.guild.me)
        if not permissions.send_messages:
            logger.warning(f"Missing send_messages permission in Ch {channel_id} (Guild {guild_id}). Cannot respond to message {message.id}.")
            return
        if not permissions.read_message_history: # Needed for context, though not strictly for responding
            logger.warning(f"Missing read_message_history permission in Ch {channel_id} (Guild {guild_id}). Context might be affected for message {message.id}.")
            # Bot might still respond, but context will be impaired.

    logger.info(f"Processing chat message ID {message.id} from {message.author.name} ({message.author.id}) in Channel {channel_id} (Guild {guild_id})")
    start_time = time.time()
    user = message.author
    user_name = user.display_name # Use display name for prompts
    
    message_content = message.content.strip() if message.content else ""
    
    thinking_msg: Optional[discord.Message] = None
    
    # Get effective settings for this context
    effective_model_name = get_guild_model(guild_id)
    effective_neko_version = get_guild_version(guild_id)
    channel_mode = get_channel_mode(guild_id, channel_id) 

    # This check is technically redundant if on_message calls this, but good safeguard
    if channel_mode is None: 
        logger.error(f"process_message called for unconfigured channel {channel_id} (Guild {guild_id}). Message ID {message.id}. This should not happen if called from on_message.")
        return

    logger.debug(f"[Guild {guild_id}/Ch {channel_id}] Effective settings for msg {message.id}: Model={effective_model_name}, Version={effective_neko_version}, Mode={channel_mode}")

    async with message.channel.typing(): # Show typing indicator
        try:
            # Process attachments
            files_for_prompt, images_for_prompt, attachment_errors, has_unsupported_attachments = await _process_attachments(message.attachments)
            
            # Send "thinking" message
            thinking_text = f"## {EMOJI_LUNA_THINKING} Neko is thinking, <@{user.id}>... {EMOJI_BARD_THINK}"
            try:
                thinking_msg = await message.channel.send(thinking_text, reference=message, mention_author=False)
            except discord.Forbidden:
                logger.error(f"Forbidden to send 'thinking' message in Ch {channel_id}. Cannot proceed with response for msg {message.id}.")
                return # Cannot send response if can't send thinking message
            except discord.HTTPException as e_thinking_http:
                logger.error(f"HTTP Error sending 'thinking' message for msg {message.id}: {e_thinking_http.status} - {e_thinking_http.text}. Proceeding without it.")
                thinking_msg = None # Continue, but streaming will start a new message.
            
            # Helper to send warnings related to processing
            async def send_processing_warning(text_to_warn: str):
                try:
                    # Reply to thinking_msg if it exists, else to original user message
                    warn_reference = thinking_msg if thinking_msg else message
                    await message.channel.send(text_to_warn, reference=warn_reference, mention_author=False)
                except discord.HTTPException as e_warn:
                    logger.error(f"Failed to send processing warning '{text_to_warn[:50]}' for msg {message.id}: {e_warn.status}", exc_info=False)

            if attachment_errors:
                await send_processing_warning("Sorry, there were some issues with attachments:\n- " + "\n- ".join(attachment_errors))
            if has_unsupported_attachments:
                await send_processing_warning("Note: Some attached file types were not supported and have been ignored.")

            prompt_file_content_str = "\n\n".join(files_for_prompt) if files_for_prompt else ""
            
            search_results_text: Optional[str] = None
            user_language_for_search_summary = "the user's language" # Default for V2

            # Determine if search is needed
            should_search = any(keyword.lower() in message_content.lower() for keyword in SEARCH_KEYWORDS) or \
                            (files_for_prompt and not images_for_prompt and not message_content) # Search if only files, no text/image

            if should_search:
                if GOOGLE_API_KEY and GOOGLE_CSE_ID:
                    search_query = message_content if message_content else "analyze the attached file(s)"
                    logger.info(f"Performing search for msg {message.id} with query: '{search_query}'")
                    keywords_for_search = extract_keywords(search_query)
                    search_results_text, search_err_msg = await google_search(keywords_for_search)
                    if not search_results_text and search_err_msg:
                        logger.info(f"Google Search failed for msg {message.id}: {search_err_msg}")
                        await send_processing_warning(f"Search Error: {search_err_msg}")
                    if effective_neko_version == "v1": # V1 always responds in Vietnamese
                        user_language_for_search_summary = "Vietnamese"
                elif any(keyword.lower() in message_content.lower() for keyword in SEARCH_KEYWORDS): # Explicit search trigger but no config
                    await send_processing_warning("Sorry, search functionality is not configured by the bot owner.")
            
            # Construct the base prompt
            base_prompt_text = ""
            if effective_neko_version == "v1":
                is_owner_message = OWNER_DISCORD_ID and str(user.id) == OWNER_DISCORD_ID
                base_prompt_text = MASTER_PROMPT_V1 if is_owner_message else PROMPT_V1
            else: # V2
                base_prompt_text = PROMPT_V2
            
            # Add channel mode specific instructions
            if channel_mode == ChannelMode.NORMAL: base_prompt_text += PROMPT_MODE_NORMAL
            elif channel_mode == ChannelMode.CODER: base_prompt_text += PROMPT_MODE_CODER
            elif channel_mode == ChannelMode.PROFESSIONAL: base_prompt_text += PROMPT_MODE_PROFESSIONAL
            
            formatted_history_str = format_context_history(channel_id, MAX_CONTEXT_TOKENS)
            history_section_str = f"\n\nConversation History:\n{formatted_history_str}" if formatted_history_str else ""
            
            # Prepare user input part of the prompt
            user_input_indicator = message_content
            if not user_input_indicator and not images_for_prompt and not files_for_prompt:
                user_input_indicator = "[User sent an empty message or only unsupported attachments]"
            if images_for_prompt: 
                user_input_indicator += " [User has attached image(s)]"
            if files_for_prompt: # Files content is added separately below
                user_input_indicator += " [User has attached file(s) - content will be provided below in the prompt]"
            
            user_input_section_str = f"User ({user_name}): {user_input_indicator}"
            
            # Combine user input and file contents for the prompt
            input_and_files_section_str = f"{user_input_section_str}\n\n{prompt_file_content_str}".strip()

            full_prompt_str = ""
            if search_results_text: # If search was performed and yielded results
                search_based_prompt = create_search_prompt(search_results_text, message_content or "analyze file/image", user_language_for_search_summary)
                full_prompt_str = f"{base_prompt_text}{history_section_str}\n\n{search_based_prompt}\n\nNeko:" # Neko's turn
            else: # Normal chat or search failed/not triggered
                full_prompt_str = f"{base_prompt_text}{history_section_str}\n\n{input_and_files_section_str}\n\nNeko:"

            logger.debug(f"Full prompt length for msg {message.id}: {len(full_prompt_str)} chars. Images: {len(images_for_prompt)}")
            
            # Get response from Gemini API
            response_stream_obj, api_error_message = await get_api_response(full_prompt_str, effective_model_name, images_for_prompt)
            
            if api_error_message:
                logger.error(f"API Error for msg {message.id}: {api_error_message}")
                error_content_for_user = f"Error: {api_error_message}"
                if thinking_msg:
                    try: await thinking_msg.edit(content=error_content_for_user[:MESSAGE_LENGTH_LIMIT])
                    except Exception as e_edit_apierr: 
                        logger.error(f"Failed to edit thinking_msg with API error for msg {message.id}: {e_edit_apierr}")
                        await send_processing_warning(error_content_for_user[:MESSAGE_LENGTH_LIMIT]) # Fallback send
                else: # If thinking_msg failed to send initially
                    await send_processing_warning(error_content_for_user[:MESSAGE_LENGTH_LIMIT])
                return # Stop processing this message

            if not response_stream_obj:
                logger.error(f"No response stream received from API for msg {message.id} (and no specific error message).")
                no_stream_err_content = "Error: Could not get a response from the AI. Please try again."
                if thinking_msg:
                    try: await thinking_msg.edit(content=no_stream_err_content)
                    except Exception as e_edit_nostream: 
                        logger.error(f"Failed to edit thinking_msg with no-stream error for msg {message.id}: {e_edit_nostream}")
                        await send_processing_warning(no_stream_err_content)
                else:
                    await send_processing_warning(no_stream_err_content)
                return

            # --- Handle Streaming Response to Discord ---
            final_response_text = await _handle_streaming_response(
                target_message=message, # User's original message
                initial_bot_message=thinking_msg, # Bot's "thinking" message
                response_stream=response_stream_obj, 
                start_time=start_time, 
                effective_model_name=effective_model_name,
                interaction=None # This is not an interaction context
            )

            # --- Update Channel Context Memory ---
            if final_response_text:
                # Store the user input that led to this response (with indicators, not full file content)
                user_context_entry_content = user_input_indicator 
                user_entry_tuple = (time.time(), user.id, user_name, "User", user_context_entry_content)
                CONTEXT_MEMORY[channel_id].append(user_entry_tuple)
                
                # Store bot's response
                bot_entry_tuple = (time.time() + 0.1, bot.user.id, bot.user.display_name, "Neko", final_response_text)
                CONTEXT_MEMORY[channel_id].append(bot_entry_tuple)
                logger.info(f"Updated context for Ch {channel_id} after msg {message.id}. New context length: {len(CONTEXT_MEMORY[channel_id])}")
            else:
                logger.warning(f"No final response text received from streaming for msg {message.id}. Context not updated with bot response.")

        # --- General Exception Handling for process_message ---
        except Exception as e_process_msg:
             logger.error(f"Unhandled error in process_message for msg {message.id}: {e_process_msg}", exc_info=True)
             unexpected_error_msg = "An unexpected internal error occurred while processing your message."
             if thinking_msg: # Try to edit the thinking message with the error
                 try: await thinking_msg.edit(content=unexpected_error_msg)
                 except Exception: pass # Suppress errors from editing here, already logged main one
             # else: No thinking_msg, error already logged. Could try sending new if critical.


# --- Command Handling Logic (Unified for Prefix & Slash) ---
async def _command_handler_logic(
    source: commands.Context | discord.Interaction, 
    content: str, # Query content from user
    is_search_command: bool, 
    attachments: Optional[List[discord.Attachment]] = None
):
    start_time = time.time()
    
    # Differentiate context (prefix vs. slash)
    user: discord.User | discord.Member
    channel: discord.abc.MessageableChannel # Covers TextChannel, DMChannel, Thread, etc.
    original_user_message: Optional[discord.Message] = None # User's invoking message (for prefix)
    interaction_obj: Optional[discord.Interaction] = None
    command_name: str = "unknown_command"
    guild_id: Optional[int] = None
    
    # Helper for sending initial response / deferring
    # For slash, this will be interaction.followup.send or interaction.edit_original_response
    # For prefix, this will be channel.send or message.edit
    send_response_func = None 
    edit_response_func = None

    if isinstance(source, commands.Context):
        user = source.author
        channel = source.channel
        original_user_message = source.message
        command_name = source.invoked_with or "prefix_command"
        guild_id = source.guild.id if source.guild else None
        send_response_func = source.send # This will be used for followups if initial send fails
        # edit_response_func will be set after thinking_msg is sent
        typing_context = channel.typing() if hasattr(channel, 'typing') else contextlib.nullcontext()
    elif isinstance(source, discord.Interaction):
        user = source.user
        channel = source.channel # Can be None if interaction is, e.g., on a button in ephemeral message
        interaction_obj = source
        command_name = source.command.name if source.command else "slash_command"
        guild_id = source.guild_id
        # Deferral handles initial response; followup for subsequent.
        # send_response_func = interaction_obj.followup.send # For new messages after initial
        # edit_response_func = interaction_obj.edit_original_response # For the deferred message
        typing_context = channel.typing() if channel and hasattr(channel, 'typing') else contextlib.nullcontext()
    else:
        logger.error(f"Invalid source type for _command_handler_logic: {type(source)}")
        return

    if not channel: # Should only happen for certain interaction types not used here, or if channel is gone
        logger.error(f"Command '{command_name}' invoked by {user.name} but channel is not available.")
        if interaction_obj and not interaction_obj.response.is_done():
            try: await interaction_obj.response.send_message("Error: Cannot process command, channel context is missing.", ephemeral=True)
            except Exception: pass
        return

    user_name = user.display_name
    channel_id = channel.id
    
    # Get effective settings
    effective_model_name = get_guild_model(guild_id)
    effective_neko_version = get_guild_version(guild_id)
    # For commands, if channel isn't configured for normal chat, default to NORMAL mode for command processing.
    channel_mode_for_command = get_channel_mode(guild_id, channel_id) or ChannelMode.NORMAL
    
    logger.debug(f"[Cmd][Gid {guild_id}/Ch {channel_id}] User: {user_name}, Cmd: {command_name}. Settings: M={effective_model_name}, V={effective_neko_version}, Mode={channel_mode_for_command}")

    thinking_msg_obj: Optional[discord.Message] = None
    
    # Use channel.typing() if available
    async with typing_context:
        try:
            # Process attachments (from original message for prefix, or passed for slash)
            attachments_to_process = attachments or (original_user_message.attachments if original_user_message else [])
            files_for_prompt, images_for_prompt, attachment_errors, has_unsupported_attachments = await _process_attachments(attachments_to_process)
            
            # Initial response: Defer for slash, send "thinking" for prefix
            if interaction_obj:
                if not interaction_obj.response.is_done(): # Check if already responded (e.g. by a check failure)
                    await interaction_obj.response.defer(ephemeral=False) # Defer public response
                thinking_msg_obj = await interaction_obj.original_response() # Get the deferred message to edit
                edit_response_func = interaction_obj.edit_original_response
                send_response_func = interaction_obj.followup.send # For subsequent messages
            elif original_user_message: # Prefix command
                thinking_text_prefix = f"## {EMOJI_LUNA_THINKING}  Neko is on it,  <@{user.id}>... {EMOJI_BARD_THINK}"
                try:
                    thinking_msg_obj = await channel.send(thinking_text_prefix, reference=original_user_message, mention_author=False)
                    edit_response_func = thinking_msg_obj.edit
                except discord.Forbidden:
                    logger.error(f"[Cmd] Forbidden to send 'thinking' message in Ch {channel_id} for cmd {command_name}. Cannot proceed.")
                    return
                except discord.HTTPException as e_thinking_cmd_http:
                    logger.error(f"[Cmd] HTTP Error sending 'thinking' message for cmd {command_name}: {e_thinking_cmd_http.status}. Proceeding without.")
                    thinking_msg_obj = None # Will cause streaming to send new message
                send_response_func = channel.send # For subsequent or new messages
            
            # Helper to send warnings
            async def send_cmd_processing_warning(text_to_warn: str):
                try:
                    if interaction_obj: # Send ephemeral followup for interactions
                        await interaction_obj.followup.send(text_to_warn, ephemeral=True)
                    elif channel: # For prefix, reply to thinking_msg or original_user_message
                        warn_ref = thinking_msg_obj if thinking_msg_obj else original_user_message
                        await channel.send(text_to_warn, reference=warn_ref, mention_author=False)
                except Exception as e_cmd_warn:
                    logger.error(f"[Cmd] Failed to send processing warning '{text_to_warn[:50]}' for cmd {command_name}: {e_cmd_warn}", exc_info=False)

            if attachment_errors:
                await send_cmd_processing_warning("Sorry, there were some issues with attachments:\n- " + "\n- ".join(attachment_errors))
            if has_unsupported_attachments:
                await send_cmd_processing_warning("Note: Some attached file types were not supported and have been ignored for this command.")

            prompt_file_content_str = "\n\n".join(files_for_prompt) if files_for_prompt else ""
            
            search_results_text: Optional[str] = None
            user_language_for_search_summary = "the user's language" # Default V2

            if is_search_command:
                if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
                    no_search_config_msg = "Search functionality is not configured by the bot owner."
                    if edit_response_func and thinking_msg_obj: # Try to edit initial message
                        try: await edit_response_func(content=no_search_config_msg)
                        except Exception: await send_cmd_processing_warning(no_search_config_msg) # Fallback
                    else: # No initial message to edit or edit func not set
                        await send_cmd_processing_warning(no_search_config_msg)
                    return # Stop if search is required but not configured
                
                search_query = content if content else "analyze the attached content"
                logger.info(f"[Cmd] Performing search for cmd {command_name} with query: '{search_query}'")
                keywords_for_search = extract_keywords(search_query)
                search_results_text, search_err_msg = await google_search(keywords_for_search)

                if not search_results_text and search_err_msg:
                    logger.info(f"[Cmd] Google Search failed for cmd {command_name}: {search_err_msg}")
                    await send_cmd_processing_warning(f"Search Error: {search_err_msg}")
                if effective_neko_version == "v1":
                    user_language_for_search_summary = "Vietnamese"
            
            # Construct base prompt
            base_prompt_text = ""
            if effective_neko_version == "v1":
                is_owner_cmd = OWNER_DISCORD_ID and str(user.id) == OWNER_DISCORD_ID
                base_prompt_text = MASTER_PROMPT_V1 if is_owner_cmd else PROMPT_V1
            else: # V2
                base_prompt_text = PROMPT_V2
            
            # Add mode instructions (using command's effective mode)
            if channel_mode_for_command == ChannelMode.NORMAL: base_prompt_text += PROMPT_MODE_NORMAL
            elif channel_mode_for_command == ChannelMode.CODER: base_prompt_text += PROMPT_MODE_CODER
            elif channel_mode_for_command == ChannelMode.PROFESSIONAL: base_prompt_text += PROMPT_MODE_PROFESSIONAL
            
            formatted_history_str = format_context_history(channel_id, MAX_CONTEXT_TOKENS)
            history_section_str = f"\n\nConversation History (from this channel):\n{formatted_history_str}" if formatted_history_str else ""
            
            # Prepare command input part of the prompt
            cmd_prefix_char = "/" if interaction_obj else bot.command_prefix
            command_indicator_str = f"User ({user_name}) used command: `{cmd_prefix_char}{command_name}`"
            if content: 
                command_indicator_str += f" with query: {content}"
            
            if not content and not images_for_prompt and not files_for_prompt:
                 command_indicator_str += " [User provided no specific query, images, or processable files for the command]"
            if images_for_prompt:
                command_indicator_str += " [User has attached image(s) with the command]"
            if files_for_prompt:
                command_indicator_str += " [User has attached file(s) with the command - content will be provided below in the prompt]"

            # Combine command indicator and file contents
            input_and_files_for_cmd_prompt_str = f"{command_indicator_str}\n\n{prompt_file_content_str}".strip()

            full_prompt_str = ""
            if is_search_command and search_results_text:
                search_based_cmd_prompt = create_search_prompt(search_results_text, content or "analyze attached content", user_language_for_search_summary)
                full_prompt_str = f"{base_prompt_text}{history_section_str}\n\n{search_based_cmd_prompt}\n\nNeko:"
            else:
                full_prompt_str = f"{base_prompt_text}{history_section_str}\n\n{input_and_files_for_cmd_prompt_str}\n\nNeko:"

            logger.debug(f"[Cmd] Full prompt length for cmd {command_name}: {len(full_prompt_str)} chars. Images: {len(images_for_prompt)}")

            response_stream_obj, api_error_message = await get_api_response(full_prompt_str, effective_model_name, images_for_prompt)

            if api_error_message:
                logger.error(f"[Cmd] API Error for cmd {command_name}: {api_error_message}")
                error_content_for_user_cmd = f"API Error for command `{command_name}`: {api_error_message}"
                if edit_response_func and thinking_msg_obj: # Try edit
                    try: await edit_response_func(content=error_content_for_user_cmd[:MESSAGE_LENGTH_LIMIT])
                    except Exception: await send_cmd_processing_warning(error_content_for_user_cmd[:MESSAGE_LENGTH_LIMIT]) # Fallback
                else: # No initial message to edit, or edit func not set (e.g. prefix thinking send failed)
                    await send_cmd_processing_warning(error_content_for_user_cmd[:MESSAGE_LENGTH_LIMIT])
                return

            if not response_stream_obj:
                logger.error(f"[Cmd] No response stream received from API for cmd {command_name} (and no specific error).")
                no_stream_cmd_err = f"Error: Could not get a response from the AI for command `{command_name}`. Please try again."
                if edit_response_func and thinking_msg_obj:
                    try: await edit_response_func(content=no_stream_cmd_err)
                    except Exception: await send_cmd_processing_warning(no_stream_cmd_err)
                else:
                    await send_cmd_processing_warning(no_stream_cmd_err)
                return

            # Determine target_message for _handle_streaming_response:
            # For prefix, it's the user's command message (original_user_message).
            # For slash, it's the thinking_msg_obj (which is original_response).
            target_message_for_streamer = original_user_message if not interaction_obj and original_user_message else thinking_msg_obj

            if not target_message_for_streamer: # Should not happen if logic above is correct
                logger.error(f"[Cmd] Critical error: target_message_for_streamer is None for cmd {command_name}. Aborting stream.")
                await send_cmd_processing_warning(f"Internal error: Could not establish a message target for streaming response of `{command_name}`.")
                return

            final_response_text = await _handle_streaming_response(
                target_message=target_message_for_streamer, 
                initial_bot_message=thinking_msg_obj, # This is the one to be edited initially
                response_stream=response_stream_obj,
                start_time=start_time,
                effective_model_name=effective_model_name,
                interaction=interaction_obj # Pass interaction object
            )

            if final_response_text:
                # Use the command indicator string for context history for commands
                user_context_entry_content_cmd = command_indicator_str 
                user_entry_tuple_cmd = (time.time(), user.id, user_name, "User", user_context_entry_content_cmd)
                CONTEXT_MEMORY[channel_id].append(user_entry_tuple_cmd)
                
                bot_entry_tuple_cmd = (time.time() + 0.1, bot.user.id, bot.user.display_name, "Neko", final_response_text)
                CONTEXT_MEMORY[channel_id].append(bot_entry_tuple_cmd)
                logger.info(f"[Cmd] Updated context for Ch {channel_id} after cmd {command_name}. New context length: {len(CONTEXT_MEMORY[channel_id])}")
            else:
                logger.warning(f"[Cmd] No final response text received from streaming for cmd {command_name}. Context not updated with bot response.")

        except Exception as e_cmd_handler: # Catch-all for _command_handler_logic
            logger.error(f"[Cmd] Unexpected error in handler for cmd '{command_name}': {e_cmd_handler}", exc_info=True)
            err_cmd_unexpected = f"An internal error occurred while processing command `{command_name}`."
            # Try to edit the thinking message or send a new one
            if edit_response_func and thinking_msg_obj:
                try: await edit_response_func(content=err_cmd_unexpected)
                except Exception: pass # Already logged main error
            elif interaction_obj: # Slash command, no thinking_msg_obj to edit or edit_func not set
                try: await interaction_obj.followup.send(err_cmd_unexpected, ephemeral=True)
                except Exception: pass
            elif channel: # Prefix command, thinking_msg failed or no edit_func
                try: await channel.send(err_cmd_unexpected, reference=original_user_message, mention_author=False)
                except Exception: pass


# --- Bot Events ---
@bot.event
async def on_ready():
    logger.info(f'Bot is ready! Logged in as: {bot.user.name} ({bot.user.id})')
    if bot.owner_id: logger.info(f'Primary Owner ID: {bot.owner_id}')
    if bot.owner_ids: logger.info(f'Owner IDs: {", ".join(map(str, bot.owner_ids))}')
    
    logger.info('--- Bot Configuration Summary ---')
    logger.info(f'Command Prefix: {bot.command_prefix}')
    logger.info(f'Default Neko Version: {DEFAULT_NEKO_VERSION.upper()}')
    logger.info(f'Default AI Model: {DEFAULT_MODEL}')
    logger.info(f'Google Search: {"ENABLED" if GOOGLE_API_KEY and GOOGLE_CSE_ID else "DISABLED"}')
    logger.info(f'Settings loaded from: {SETTINGS_FILE} (for {len(guild_settings)} guilds initially)')
    logger.info('Normal chat in channels requires configuration via /setchannel or neko!setchannel.')
    logger.info('The bot will respond to DMs by default (Normal mode).')

    # Ensure guilds the bot is in have basic settings
    guilds_in_memory_settings = set(guild_settings.keys()) # These are string IDs
    current_bot_guild_ids = {str(g.id) for g in bot.guilds}
    
    needs_settings_save = False
    # Add default settings for guilds joined while bot was offline / not in settings
    for gid_str in current_bot_guild_ids:
        if gid_str not in guilds_in_memory_settings:
            logger.info(f"Guild {gid_str} found, but not in loaded settings. Creating default settings.")
            guild_settings[gid_str] = {"model": DEFAULT_MODEL, "version": DEFAULT_NEKO_VERSION, "channel_settings": {}}
            needs_settings_save = True
        else: # Guild is in settings, ensure model and version keys exist
            if "model" not in guild_settings[gid_str]:
                logger.info(f"Guild {gid_str} missing 'model' in settings. Adding default: {DEFAULT_MODEL}.")
                guild_settings[gid_str]["model"] = DEFAULT_MODEL
                needs_settings_save = True
            if "version" not in guild_settings[gid_str]:
                logger.info(f"Guild {gid_str} missing 'version' in settings. Adding default: {DEFAULT_NEKO_VERSION}.")
                guild_settings[gid_str]["version"] = DEFAULT_NEKO_VERSION
                needs_settings_save = True
            if "channel_settings" not in guild_settings[gid_str]: # Ensure channel_settings dict exists
                guild_settings[gid_str]["channel_settings"] = {}
                needs_settings_save = True


    if needs_settings_save:
        save_settings()

    # Sync slash commands
    try:
        synced_commands = await bot.tree.sync()
        logger.info(f"Successfully synced {len(synced_commands)} application (slash) commands.")
    except Exception as e_sync:
        logger.error(f"Failed to sync slash commands: {e_sync}", exc_info=True)

    # Set bot presence
    try:
        activity = discord.Activity(type=discord.ActivityType.listening, name=f"/help | {bot.command_prefix}help")
        await bot.change_presence(status=discord.Status.online, activity=activity)
        logger.info("Bot presence updated successfully.")
    except Exception as e_presence:
        logger.error(f"Failed to update bot presence: {e_presence}", exc_info=True)

@bot.event
async def on_guild_join(guild: discord.Guild):
    logger.info(f"Joined new guild: {guild.name} (ID: {guild.id})")
    guild_id_str = str(guild.id)
    
    if guild_id_str not in guild_settings:
        logger.info(f"Creating default settings for newly joined guild {guild_id_str}.")
        guild_settings[guild_id_str] = {
            "model": DEFAULT_MODEL, 
            "version": DEFAULT_NEKO_VERSION,
            "channel_settings": {} # Initialize with empty channel settings
        }
        save_settings()
    else: # Guild re-joined or was already in settings file? Ensure defaults are present.
        logger.warning(f"Guild {guild_id_str} ({guild.name}) re-joined or was already in settings. Verifying essential keys.")
        if "model" not in guild_settings[guild_id_str]:
            guild_settings[guild_id_str]["model"] = DEFAULT_MODEL
        if "version" not in guild_settings[guild_id_str]:
            guild_settings[guild_id_str]["version"] = DEFAULT_NEKO_VERSION
        if "channel_settings" not in guild_settings[guild_id_str]:
             guild_settings[guild_id_str]["channel_settings"] = {}
        save_settings() # Save if any modifications were made

@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from self or other bots
    if message.author == bot.user or message.author.bot:
        return

    # Handle command prefix
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Process normal chat messages if channel is configured or it's a DM
    guild_id = message.guild.id if message.guild else None
    # For DMs (guild_id is None), get_channel_mode returns ChannelMode.NORMAL
    # For guild channels, it returns configured mode or None
    channel_mode = get_channel_mode(guild_id, message.channel.id) 

    if channel_mode is not None: # True for DMs or configured guild channels
        if message.content or message.attachments: # Only process if there's content or attachments
            # Create a task to handle message processing asynchronously
            asyncio.create_task(process_message(message), name=f"ProcessMessageTask-{message.id}")
        # else: logger.debug(f"Ignoring empty message from {message.author.name} in Ch {message.channel.id}")
    # else: logger.debug(f"Ignoring message in Ch {message.channel.id} as it's not configured for Neko chat.")


# --- Bot Commands ---

# Chat Command (Prefix and Slash)
@bot.command(name='chat', aliases=['c'], help='Chat with Neko. Provide a message or attach files/images.')
async def chat_command_prefix(ctx: commands.Context, *, message_content: str = ""):
    if not message_content.strip() and not ctx.message.attachments:
        await ctx.send("Please provide a message or attach a file/image to chat.", reference=ctx.message, mention_author=False)
        return
    logger.info(f"Prefix command 'chat' invoked by {ctx.author.name} in Ch {ctx.channel.id}.")
    asyncio.create_task(
        _command_handler_logic(ctx, message_content.strip(), is_search_command=False, attachments=ctx.message.attachments),
        name=f"CmdHandler-ChatPrefix-{ctx.message.id}"
    )

@bot.tree.command(name="chat", description="Chat with Neko. You can attach a file or image.")
@app_commands.describe(message="Your message to Neko.", attachment="Optional: Attach a file or image.")
async def chat_command_slash(interaction: discord.Interaction, message: str = "", attachment: Optional[discord.Attachment] = None):
    attachments_list = [attachment] if attachment else []
    if not message.strip() and not attachments_list:
        await interaction.response.send_message("Please provide a message or attach a file/image to chat.", ephemeral=True)
        return
    logger.info(f"Slash command 'chat' invoked by {interaction.user.name} in Ch {interaction.channel_id}.")
    asyncio.create_task(
        _command_handler_logic(interaction, message.strip(), is_search_command=False, attachments=attachments_list),
        name=f"CmdHandler-ChatSlash-{interaction.id}"
    )

# Search Command (Prefix and Slash)
@bot.command(name='search', aliases=['s', 'find', 'google', 'tìm'], help='Search the web using Google and get a summary from Neko.')
async def search_command_prefix(ctx: commands.Context, *, query: str = ""):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        await ctx.send("Search functionality is not configured by the bot owner.", reference=ctx.message, mention_author=False)
        return
    if not query.strip() and not ctx.message.attachments: # Require query or attachment for search context
        await ctx.send("Please provide a search query or attach a file/image to search about.", reference=ctx.message, mention_author=False)
        return
    logger.info(f"Prefix command 'search' invoked by {ctx.author.name} with query: '{query[:50]}...'")
    asyncio.create_task(
        _command_handler_logic(ctx, query.strip(), is_search_command=True, attachments=ctx.message.attachments),
        name=f"CmdHandler-SearchPrefix-{ctx.message.id}"
    )

@bot.tree.command(name="search", description="Search the web with Google and get Neko's summary.")
@app_commands.describe(query="Your search query.", attachment="Optional: Attach a file/image to search about.")
async def search_command_slash(interaction: discord.Interaction, query: str = "", attachment: Optional[discord.Attachment] = None):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        await interaction.response.send_message("Search functionality is not configured by the bot owner.", ephemeral=True)
        return
    attachments_list = [attachment] if attachment else []
    if not query.strip() and not attachments_list:
        await interaction.response.send_message("Please provide a search query or attach a file/image.", ephemeral=True)
        return
    logger.info(f"Slash command 'search' invoked by {interaction.user.name} with query: '{query[:50]}...'")
    asyncio.create_task(
        _command_handler_logic(interaction, query.strip(), is_search_command=True, attachments=attachments_list),
        name=f"CmdHandler-SearchSlash-{interaction.id}"
    )

# Clear Context Command
@bot.command(name='clear', aliases=['cls', 'clean'], help='Clears Neko\'s conversation history for this channel.')
@commands.bot_has_permissions(send_messages=True) # Basic permission
async def clear_context_command_prefix(ctx: commands.Context):
    channel_id = ctx.channel.id
    if channel_id in CONTEXT_MEMORY:
        CONTEXT_MEMORY[channel_id].clear()
        logger.info(f"Cleared context for Channel {channel_id} by {ctx.author.name} (prefix command).")
        await ctx.send("Neko's conversation history for this channel has been cleared.", reference=ctx.message, mention_author=False)
    else:
        await ctx.send("There is no conversation history stored for this channel to clear.", reference=ctx.message, mention_author=False)

@bot.tree.command(name="clear", description="Clears Neko's conversation history for this channel.")
async def clear_context_command_slash(interaction: discord.Interaction):
    if not interaction.channel_id: # Should always be present for channel-based commands
        await interaction.response.send_message("Error: Cannot determine channel ID.", ephemeral=True)
        return
    channel_id = interaction.channel_id
    if channel_id in CONTEXT_MEMORY:
        CONTEXT_MEMORY[channel_id].clear()
        logger.info(f"Cleared context for Channel {channel_id} by {interaction.user.name} (slash command).")
        await interaction.response.send_message("Neko's conversation history for this channel has been cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("There is no conversation history stored for this channel to clear.", ephemeral=True)

# --- Admin/Owner Commands ---

@bot.command(name='setmodel', help=f'Set the AI model for this server (Bot Owner only). Available: {", ".join(AVAILABLE_MODELS)}')
@commands.is_owner() 
@commands.guild_only() # This command is guild-specific
async def setmodel_command_prefix(ctx: commands.Context, *, model_name: str):
    guild_id_str = str(ctx.guild.id)
    requested_model = model_name.strip().lower()
    
    found_model: Optional[str] = None
    if requested_model in AVAILABLE_MODELS:
        found_model = requested_model
    else: # Try partial match if exact not found
        matches = [m for m in AVAILABLE_MODELS if requested_model in m]
        if len(matches) == 1:
            found_model = matches[0]
            logger.info(f"Partial match for model '{requested_model}' found: '{found_model}'. Assuming this is intended.")
        elif len(matches) > 1:
            await ctx.send(f"Ambiguous model name '{requested_model}'. It matches multiple available models: {', '.join(f'`{m}`' for m in matches)}. Please be more specific.", reference=ctx.message, mention_author=False)
            return
            
    if found_model:
        guild_settings.setdefault(guild_id_str, {})["model"] = found_model
        save_settings()
        logger.info(f"[Guild {guild_id_str}] AI model set to '{found_model}' by owner {ctx.author.name} (prefix command).")
        await ctx.send(f"AI model for this server has been set to `{found_model}`.", reference=ctx.message, mention_author=False)
    else:
        models_list_str = "\n- ".join([f"`{m}`" for m in AVAILABLE_MODELS])
        await ctx.send(f"Invalid model name: `{requested_model}`. Please choose from the available models:\n- {models_list_str}", reference=ctx.message, mention_author=False)

@setmodel_command_prefix.error
async def setmodel_prefix_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.NotOwner):
        await ctx.send("This command can only be used by the bot owner.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Usage: `{ctx.prefix}setmodel <model_name>`\nSee `neko!help` for available models.", reference=ctx.message, mention_author=False)
    else:
        logger.error(f"Error in prefix command 'setmodel': {error}", exc_info=True)
        await ctx.send("An unexpected error occurred while trying to set the model.", reference=ctx.message, mention_author=False)

@bot.tree.command(name="setmodel", description="Set the AI model for this server (Bot Owner only).")
@app_commands.describe(model="Choose the AI model to use for this server.")
@app_commands.choices(model=[app_commands.Choice(name=m, value=m) for m in AVAILABLE_MODELS])
@app_commands.guild_only()
# @app_commands.checks.is_owner() # discord.py's built-in check for slash commands
async def setmodel_command_slash(interaction: discord.Interaction, model: app_commands.Choice[str]):
    # Manual owner check for slash commands as @commands.is_owner() doesn't directly apply to app_commands' callback
    if not await bot.is_owner(interaction.user):
        await interaction.response.send_message("This command can only be used by the bot owner.", ephemeral=True)
        return

    guild_id_str = str(interaction.guild_id)
    chosen_model_value = model.value # Value from the Choice object
    
    if chosen_model_value in AVAILABLE_MODELS:
        guild_settings.setdefault(guild_id_str, {})["model"] = chosen_model_value
        save_settings()
        logger.info(f"[Guild {guild_id_str}] AI model set to '{chosen_model_value}' by owner {interaction.user.name} (slash command).")
        await interaction.response.send_message(f"AI model for this server has been set to `{chosen_model_value}`.", ephemeral=True)
    else: # Should not happen if choices are sourced from AVAILABLE_MODELS, but defensive
        await interaction.response.send_message(f"Error: Invalid model selected (`{chosen_model_value}`). This should not happen.", ephemeral=True)


@bot.command(name='toggleversion', help='Toggle Neko\'s V1 (Vietnamese-focused) or V2 (Multilingual Sage) personality for this server (Bot Owner only).')
@commands.is_owner()
@commands.guild_only()
async def toggleversion_command_prefix(ctx: commands.Context):
    guild_id_str = str(ctx.guild.id)
    guild_specific_settings = guild_settings.setdefault(guild_id_str, {})
    current_version = guild_specific_settings.get("version", DEFAULT_NEKO_VERSION)
    
    new_version = "v1" if current_version == "v2" else "v2"
    guild_specific_settings["version"] = new_version
    save_settings()
    
    feedback_message = f"Neko's personality for this server has been switched to **{new_version.upper()}**."
    if new_version == "v1":
        feedback_message += " (Focus: Vietnamese, playful assistant)"
    else: # V2
        feedback_message += " (Focus: Multilingual, wise sage catgirl)"
        
    logger.info(f"[Guild {guild_id_str}] Neko version toggled to '{new_version.upper()}' by owner {ctx.author.name} (prefix command).")
    await ctx.send(feedback_message, reference=ctx.message, mention_author=False)

@toggleversion_command_prefix.error
async def toggleversion_prefix_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.NotOwner):
        await ctx.send("This command can only be used by the bot owner.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.", reference=ctx.message, mention_author=False)
    else:
        logger.error(f"Error in prefix command 'toggleversion': {error}", exc_info=True)
        await ctx.send("An unexpected error occurred while toggling Neko's version.", reference=ctx.message, mention_author=False)

@bot.tree.command(name="toggleversion", description="Switch Neko V1/V2 personality for this server (Bot Owner only).")
@app_commands.guild_only()
async def toggleversion_command_slash(interaction: discord.Interaction):
    if not await bot.is_owner(interaction.user): # Manual owner check
        await interaction.response.send_message("This command can only be used by the bot owner.", ephemeral=True)
        return

    guild_id_str = str(interaction.guild_id)
    guild_specific_settings = guild_settings.setdefault(guild_id_str, {})
    current_version = guild_specific_settings.get("version", DEFAULT_NEKO_VERSION)
    
    new_version = "v1" if current_version == "v2" else "v2"
    guild_specific_settings["version"] = new_version
    save_settings()

    feedback_message = f"Neko's personality for this server has been switched to **{new_version.upper()}**."
    if new_version == "v1":
        feedback_message += " (Focus: Vietnamese, playful assistant)"
    else: # V2
        feedback_message += " (Focus: Multilingual, wise sage catgirl)"

    logger.info(f"[Guild {guild_id_str}] Neko version toggled to '{new_version.upper()}' by owner {interaction.user.name} (slash command).")
    await interaction.response.send_message(feedback_message, ephemeral=True)


# Channel Configuration Commands
@bot.command(name='setchannel', help=f'Configure a channel for Neko to respond in normal chat, with a specific mode. Modes: {", ".join(ChannelMode.ALL_MODES)}.')
@commands.has_permissions(manage_channels=True) 
@commands.bot_has_permissions(send_messages=True)
@commands.guild_only()
async def setchannel_prefix(ctx: commands.Context, channel: discord.TextChannel, mode: str):
    guild_id_str = str(ctx.guild.id)
    channel_id_str = str(channel.id)
    normalized_mode = mode.strip().lower()

    if normalized_mode not in ChannelMode.ALL_MODES:
        valid_modes_str = ", ".join(f"`{m}`" for m in ChannelMode.ALL_MODES)
        await ctx.send(f"Invalid mode: `{normalized_mode}`. Please use one of: {valid_modes_str}.", reference=ctx.message, mention_author=False)
        return
    
    # Ensure guild entry and channel_settings sub-dictionary exist
    guild_data = guild_settings.setdefault(guild_id_str, {"model": DEFAULT_MODEL, "version": DEFAULT_NEKO_VERSION, "channel_settings": {}})
    if "channel_settings" not in guild_data: # Should be created by setdefault if guild was new, but defensive
        guild_data["channel_settings"] = {}
    
    guild_data["channel_settings"][channel_id_str] = {"mode": normalized_mode}
    save_settings()
    
    logger.info(f"[Guild {guild_id_str}] Channel {channel.name} (ID: {channel_id_str}) set to mode '{normalized_mode}' by {ctx.author.name} (prefix command).")
    await ctx.send(f"Channel {channel.mention} is now configured for Neko chat in **{normalized_mode}** mode. Neko will respond to normal messages here.", reference=ctx.message, mention_author=False)

@setchannel_prefix.error
async def setchannel_prefix_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Usage: `{ctx.prefix}setchannel #channel <mode>`. Valid modes: {', '.join(f'`{m}`' for m in ChannelMode.ALL_MODES)}.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.ChannelNotFound):
        await ctx.send(f"Could not find the channel: `{error.argument}`. Please make sure it's a valid text channel in this server.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.BadArgument): # Catches if mode wasn't turned into string, or channel was wrong type
        await ctx.send(f"Invalid channel or mode provided. Please specify a text channel and a valid mode.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You need the `Manage Channels` permission to use this command.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.", reference=ctx.message, mention_author=False)
    else:
        logger.error(f"Error in prefix command 'setchannel': {error}", exc_info=True)
        await ctx.send("An unexpected error occurred while setting the channel configuration.", reference=ctx.message, mention_author=False)

@bot.tree.command(name="setchannel", description="Configure a channel for Neko chat with a specific mode.")
@app_commands.describe(channel="The text channel to configure for Neko.", mode="The chat mode for Neko in this channel.")
@app_commands.choices(mode=[app_commands.Choice(name=m.capitalize(), value=m) for m in ChannelMode.ALL_MODES])
@app_commands.checks.has_permissions(manage_channels=True)
@app_commands.guild_only()
async def setchannel_slash(interaction: discord.Interaction, channel: discord.TextChannel, mode: app_commands.Choice[str]):
    guild_id_str = str(interaction.guild_id)
    channel_id_str = str(channel.id)
    chosen_mode_value = mode.value # Value from the Choice object

    guild_data = guild_settings.setdefault(guild_id_str, {"model": DEFAULT_MODEL, "version": DEFAULT_NEKO_VERSION, "channel_settings": {}})
    if "channel_settings" not in guild_data:
        guild_data["channel_settings"] = {}
        
    guild_data["channel_settings"][channel_id_str] = {"mode": chosen_mode_value}
    save_settings()
    
    logger.info(f"[Guild {guild_id_str}] Channel {channel.name} (ID: {channel_id_str}) set to mode '{chosen_mode_value}' by {interaction.user.name} (slash command).")
    await interaction.response.send_message(f"Channel {channel.mention} is now configured for Neko chat in **{chosen_mode_value}** mode. Neko will respond to normal messages here.", ephemeral=True)


@bot.command(name='removechannel', aliases=['rmchannel', 'disablechannel'], help='Stop Neko from responding to normal chat in a configured channel.')
@commands.has_permissions(manage_channels=True)
@commands.bot_has_permissions(send_messages=True)
@commands.guild_only()
async def removechannel_prefix(ctx: commands.Context, channel: discord.TextChannel):
    guild_id_str = str(ctx.guild.id)
    channel_id_str = str(channel.id)
    
    guild_data = guild_settings.get(guild_id_str)
    if not guild_data or "channel_settings" not in guild_data or channel_id_str not in guild_data["channel_settings"]:
        await ctx.send(f"Channel {channel.mention} is not currently configured for Neko chat.", reference=ctx.message, mention_author=False)
        return
        
    del guild_data["channel_settings"][channel_id_str]
    # Clean up empty channel_settings dict if it becomes empty
    if not guild_data["channel_settings"]:
        del guild_data["channel_settings"] 
    save_settings()
    
    logger.info(f"[Guild {guild_id_str}] Configuration removed for Ch {channel.name} (ID: {channel_id_str}) by {ctx.author.name} (prefix command).")
    await ctx.send(f"Neko will no longer respond to normal messages in {channel.mention}. (Commands will still work).", reference=ctx.message, mention_author=False)

@removechannel_prefix.error
async def removechannel_prefix_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Usage: `{ctx.prefix}removechannel #channel_name`", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.ChannelNotFound):
        await ctx.send(f"Could not find the channel: `{error.argument}`.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.BadArgument):
        await ctx.send(f"Invalid channel provided. Please specify a text channel.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You need the `Manage Channels` permission to use this command.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.", reference=ctx.message, mention_author=False)
    else:
        logger.error(f"Error in prefix command 'removechannel': {error}", exc_info=True)
        await ctx.send("An unexpected error occurred while trying to remove channel configuration.", reference=ctx.message, mention_author=False)

@bot.tree.command(name="removechannel", description="Stop Neko responding to normal chat in a channel.")
@app_commands.describe(channel="The channel to remove Neko's chat configuration from.")
@app_commands.checks.has_permissions(manage_channels=True)
@app_commands.guild_only()
async def removechannel_slash(interaction: discord.Interaction, channel: discord.TextChannel):
    guild_id_str = str(interaction.guild_id)
    channel_id_str = str(channel.id)
    
    guild_data = guild_settings.get(guild_id_str)
    # Check more directly if channel_id_str is in the nested dictionary
    if not guild_data or "channel_settings" not in guild_data or channel_id_str not in guild_data.get("channel_settings", {}):
        await interaction.response.send_message(f"Channel {channel.mention} is not currently configured for Neko chat.", ephemeral=True)
        return
        
    del guild_data["channel_settings"][channel_id_str]
    if not guild_data["channel_settings"]: # If no channels are configured anymore
        del guild_data["channel_settings"]
    save_settings()
    
    logger.info(f"[Guild {guild_id_str}] Configuration removed for Ch {channel.name} (ID: {channel_id_str}) by {interaction.user.name} (slash command).")
    await interaction.response.send_message(f"Neko will no longer respond to normal messages in {channel.mention}. (Commands will still work).", ephemeral=True)


@bot.command(name='listchannels', aliases=['lschannels', 'showchannels'], help='List channels configured for Neko chat and their modes in this server.')
@commands.bot_has_permissions(send_messages=True, embed_links=True) # Embed links for richer output
@commands.guild_only()
async def listchannels_prefix(ctx: commands.Context):
    guild_id_str = str(ctx.guild.id)
    # Get channel settings safely, defaulting to empty dict if not found
    channel_settings_map = guild_settings.get(guild_id_str, {}).get("channel_settings", {}) 
    
    embed = discord.Embed(
        title=f"⚙️ Neko Chat Configurations for {ctx.guild.name}", 
        color=discord.Color.from_rgb(177, 156, 217) # A Neko-ish purple
    )
    
    if not channel_settings_map:
        embed.description = f"No channels are currently configured for Neko to respond to normal chat messages.\nUse `{ctx.prefix}setchannel #channel <mode>` to configure one."
    else:
        description_lines = ["Here are the channels where Neko is active for normal chat:"]
        for cid_str, channel_data in channel_settings_map.items():
            mode = channel_data.get("mode", "Unknown Mode")
            channel_obj = ctx.guild.get_channel(int(cid_str)) # Attempt to get channel object
            
            channel_mention = f"`{cid_str}` (Channel not found or inaccessible)"
            if channel_obj:
                channel_mention = channel_obj.mention
            
            description_lines.append(f"- {channel_mention}: **{mode.capitalize()}** mode")
        embed.description = "\n".join(description_lines)
        embed.set_footer(text=f"Use `{ctx.prefix}setchannel` or `{ctx.prefix}removechannel` to manage configurations.")
        
    await ctx.send(embed=embed, reference=ctx.message, mention_author=False)

@listchannels_prefix.error
async def listchannels_prefix_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.BotMissingPermissions):
        perms_needed = ", ".join(f"`{p}`" for p in error.missing_permissions)
        await ctx.send(f"I'm missing permissions to do that: {perms_needed}. Please grant them and try again.", reference=ctx.message, mention_author=False)
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.", reference=ctx.message, mention_author=False)
    else:
        logger.error(f"Error in prefix command 'listchannels': {error}", exc_info=True)
        await ctx.send("An unexpected error occurred while listing configured channels.", reference=ctx.message, mention_author=False)

@bot.tree.command(name="listchannels", description="List channels configured for Neko chat and their modes.")
@app_commands.guild_only()
async def listchannels_slash(interaction: discord.Interaction):
    if not interaction.guild: # Should be guaranteed by @app_commands.guild_only()
        await interaction.response.send_message("Error: This command must be used in a server.", ephemeral=True)
        return
        
    guild_id_str = str(interaction.guild_id)
    channel_settings_map = guild_settings.get(guild_id_str, {}).get("channel_settings", {})
    
    embed = discord.Embed(
        title=f"⚙️ Neko Chat Configurations for {interaction.guild.name}",
        color=discord.Color.from_rgb(177, 156, 217)
    )
    
    if not channel_settings_map:
        embed.description = "No channels are currently configured for Neko to respond to normal chat messages.\nUse `/setchannel` to configure one."
    else:
        description_lines = ["Here are the channels where Neko is active for normal chat:"]
        for cid_str, channel_data in channel_settings_map.items():
            mode = channel_data.get("mode", "Unknown Mode")
            channel_obj = interaction.guild.get_channel(int(cid_str))
            
            channel_mention = f"`{cid_str}` (Channel not found or inaccessible)"
            if channel_obj:
                channel_mention = channel_obj.mention
                
            description_lines.append(f"- {channel_mention}: **{mode.capitalize()}** mode")
        embed.description = "\n".join(description_lines)
        embed.set_footer(text="Use `/setchannel` or `/removechannel` to manage configurations.")
        
    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- Help Command ---
@bot.command(name='help', aliases=['h', 'info', 'halp'], help='Shows this help message with Neko\'s commands and server settings.')
async def help_command_prefix(ctx: commands.Context, *, command_name: Optional[str] = None):
    # command_name arg can be used later for specific command help if desired
    await send_help_embed(ctx)

@bot.tree.command(name="help", description="Shows help information, commands, and current server settings for Neko.")
async def help_command_slash(interaction: discord.Interaction):
    await send_help_embed(interaction)

async def send_help_embed(source: commands.Context | discord.Interaction):
    user = source.user if isinstance(source, discord.Interaction) else source.author
    guild = source.guild
    guild_id = guild.id if guild else None
    
    eff_model = get_guild_model(guild_id)
    eff_ver = get_guild_version(guild_id)
    
    # Format available models for display
    models_list_str = "\n- ".join([f"`{m}`" for m in AVAILABLE_MODELS])
    models_list_str = "\n- " + models_list_str if models_list_str else " (No models listed as available)"

    embed = discord.Embed(
        title=f"{EMOJI_NEKO_EARS} Neko Bot Help & Information",
        color=discord.Color.from_rgb(227, 190, 255), # Light Neko purple
        description=f"Hello {user.mention}! I'm Neko, your charming and intelligent catgirl assistant! Here's how you can interact with me."
    )
    
    if bot.user and bot.user.avatar:
        embed.set_thumbnail(url=bot.user.avatar.url)
    
    # Server-specific settings (if in a guild)
    if guild:
        embed.add_field(
            name=f"⚙️ Settings for {guild.name}", 
            value=f"**Personality Version:** Neko {eff_ver.upper()}\n"
                  f"**AI Model:** `{eff_model}`", 
            inline=False
        )
        embed.add_field(
            name="💬 Normal Chat Activation",
            value=f"Neko responds to normal messages in channels configured via `/setchannel` or `{bot.command_prefix}setchannel`.\n"
                  f"Use `/listchannels` or `{bot.command_prefix}listchannels` to see currently active channels and their modes.",
            inline=False
        )
    else: # DM context
        embed.add_field(
            name="💬 Normal Chat",
            value="Neko is active in DMs by default (using Normal mode and default settings).",
            inline=False
        )

    # General Commands
    embed.add_field(
        name="🤖 General Commands",
        value=f"- `{bot.command_prefix}chat` or `/chat` - Chat with me.\n"
              f"- `{bot.command_prefix}search` or `/search` - Ask me to search the web.\n"
              f"- `{bot.command_prefix}clear` or `/clear` - Clear my memory of this channel's chat.\n"
              f"- `{bot.command_prefix}help` or `/help` - Show this message.",
        inline=False
    )
    
    # Channel Configuration Commands (for users with Manage Channels perm)
    embed.add_field(
        name="🔧 Channel Configuration (Requires `Manage Channels` Permission)",
        value=f"- `{bot.command_prefix}setchannel #channel <mode>` or `/setchannel` - Enable Neko chat in a channel.\n"
              f"- `{bot.command_prefix}removechannel #channel` or `/removechannel` - Disable Neko chat.\n"
              f"- `{bot.command_prefix}listchannels` or `/listchannels` - See configured channels.",
        inline=False
    )
    embed.add_field(
        name="✨ Channel Modes (for `setchannel`)",
        value=f"- `{ChannelMode.NORMAL}`: Standard, balanced Neko.\n"
              f"- `{ChannelMode.CODER}`: Neko the expert programmer.\n"
              f"- `{ChannelMode.PROFESSIONAL}`: Neko the analytical expert.",
        inline=False
    )
    
    # Bot Owner Commands
    embed.add_field(
        name="👑 Bot Owner Commands",
        value=f"- `{bot.command_prefix}setmodel <model_name>` or `/setmodel` - Change AI model for the server.\n"
              f"- `{bot.command_prefix}toggleversion` or `/toggleversion` - Switch Neko V1/V2 personality.",
        inline=False
    )
    
    embed.add_field(
        name="🧠 AI Models",
        value=f"**Currently using:** `{eff_model}` for this context.\n"
              f"**Available models for owner to set:** {models_list_str}",
        inline=False
    )
    embed.add_field(
        name="📎 Attachments",
        value="You can attach images (JPEG, PNG, GIF, WebP) and common text/code files when chatting or using commands. I'll do my best to understand them!",
        inline=False
    )
    
    footer_text = f"My prefix is: {bot.command_prefix}"
    if guild:
        footer_text += f" | Settings shown are for {guild.name}."
    else:
        footer_text += " | Settings shown are defaults for DMs."
    embed.set_footer(text=footer_text)
    embed.timestamp = datetime.now(timezone.utc)
    
    try:
        if isinstance(source, discord.Interaction):
            if not source.response.is_done():
                await source.response.send_message(embed=embed, ephemeral=True)
            else: # If already deferred or responded, use followup
                await source.followup.send(embed=embed, ephemeral=True)
        else: # commands.Context
            await source.send(embed=embed)
    except discord.Forbidden:
        logger.warning(f"Cannot send help embed in channel {source.channel.id if source.channel else 'DM'} due to Forbidden error.")
    except discord.HTTPException as e_help_http:
        logger.error(f"HTTP error sending help embed: {e_help_http.status} - {e_help_http.text}", exc_info=True)


# --- Global Slash Command Error Handler ---
'''@bot.tree.on_error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    original_error = getattr(error, 'original', error) # Get original error if wrapped
    command_name = interaction.command.name if interaction.command else "unknown_slash_command"

    # Log chi tiết lỗi trước
    logger.error(f"AppCommandError encountered for command '{command_name}' by user {interaction.user.name} (ID: {interaction.user.id}). Error Type: {type(error).__name__}, Original Error Type: {type(original_error).__name__}, Error: {error}", exc_info=True)


    if isinstance(error, app_commands.CheckFailure): # Permissions, owner checks, etc.
        user_message = "You do not have the necessary permissions or conditions to use this command here."
        if isinstance(original_error, commands.NotOwner):
            user_message = "This command can only be used by the bot owner."
        elif isinstance(error, app_commands.MissingPermissions):
            missing_perms_str = ', '.join(f"`{perm}`" for perm in error.missing_permissions)
            user_message = f"You are missing the required permissions to use this command: {missing_perms_str}."
        elif isinstance(error, app_commands.NoPrivateMessage): # Guild only command used in DM
            user_message = "This command can only be used in a server."

        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(user_message, ephemeral=True)
            else:
                await interaction.followup.send(user_message, ephemeral=True)
        except discord.HTTPException as e_resp:
            logger.error(f"Failed to send CheckFailure response for '{command_name}': {e_resp}")
        logger.warning(f"Slash command '{command_name}' check failed for user {interaction.user.name}: {type(original_error).__name__}")

    elif isinstance(error, app_commands.CommandInvokeError): # Error raised within the command's code
        # original_error is the actual exception raised by your command code
        logger.error(f"Error invoking slash command '{command_name}' by user {interaction.user.name}. Original error: {original_error}", exc_info=original_error) # Log with original error's traceback
        error_msg_for_user = f"An error occurred while running the command `/{command_name}`: `{type(original_error).__name__}`. The bot owner has been notified."
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(error_msg_for_user[:1900], ephemeral=True)
            else: # Already responded (e.g., deferred), use followup
                await interaction.followup.send(error_msg_for_user[:1900], ephemeral=True)
        except discord.HTTPException as e_report_invoke_err:
            logger.error(f"Failed to report CommandInvokeError for '{command_name}' to user: {e_report_invoke_err}")
    
    elif isinstance(error, app_commands.CommandNotFound):
        logger.warning(f"Unknown slash command invoked by {interaction.user.name}: {interaction.data.get('name', 'N/A')}")
        # Discord usually handles this, but good to be aware
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("Sorry, I don't recognize that command.", ephemeral=True)
        except discord.HTTPException as e_resp:
             logger.error(f"Failed to send CommandNotFound response for '{command_name}': {e_resp}")


    elif isinstance(error, app_commands.TransformerError):
        logger.warning(f"TransformerError for slash command '{command_name}' by {interaction.user.name}. Parameter: {error.param.name if error.param else 'Unknown'}, Value: '{error.value}', Type: {error.type}. Error: {error}", exc_info=True)
        user_message = f"There was an issue with the value you provided for the option `{error.param.name if error.param else 'one of the options'}`. Please check and try again."
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(user_message, ephemeral=True)
            else:
                await interaction.followup.send(user_message, ephemeral=True)
        except discord.HTTPException as e_resp:
            logger.error(f"Failed to send TransformerError response for '{command_name}': {e_resp}")

    elif isinstance(error, app_commands.CommandOnCooldown):
        logger.info(f"Slash command '{command_name}' on cooldown for user {interaction.user.name}. Retry after: {error.retry_after:.2f}s")
        user_message = f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds."
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(user_message, ephemeral=True)
            else:
                await interaction.followup.send(user_message, ephemeral=True)
        except discord.HTTPException as e_resp:
             logger.error(f"Failed to send CommandOnCooldown response for '{command_name}': {e_resp}")

    else: # Other AppCommandError types or unexpected errors
        logger.error(f"Unhandled App Command Error for '{command_name}' by user {interaction.user.name}. Error Type: {type(error).__name__}, Error: {error}", exc_info=True)
        user_message = "An unexpected error occurred with this command. Please try again later, or contact the bot owner if the issue persists."
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(user_message, ephemeral=True)
            else:
                await interaction.followup.send(user_message, ephemeral=True)
        except discord.HTTPException as e_resp:
             logger.error(f"Failed to send generic error response for '{command_name}': {e_resp}")
'''

# --- Main Execution & Cleanup ---
async def main():
    if not DISCORD_BOT_TOKEN:
        logger.critical("CRITICAL: DISCORD_BOT_TOKEN environment variable not set. Bot cannot start.")
        print("CRITICAL: DISCORD_BOT_TOKEN environment variable not set. Bot cannot start.")
        os._exit(1) # Hard exit if token is missing
    
    try:
        logger.info("Neko Bot is starting up...")
        # Initial settings load is already done globally
        await bot.start(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.critical("CRITICAL: Discord login failed. Please check the bot token.", exc_info=True)
        print("CRITICAL: Discord login failed. Is the token correct and valid?")
        os._exit(1)
    except discord.PrivilegedIntentsRequired as e_intents:
        logger.critical(f"CRITICAL: Missing privileged intents: {e_intents}. Please enable them in the Discord Developer Portal.", exc_info=True)
        print(f"CRITICAL: Missing privileged intents: {e_intents}. Enable them in your bot's settings on the Discord Developer Portal.")
        os._exit(1)
    except Exception as e_main_runtime: # Catch any other exception during startup or runtime
        logger.critical(f"CRITICAL: An unexpected error occurred during bot execution: {e_main_runtime}", exc_info=True)
        # Using sys.exit() allows the finally block in __main__ to run cleanup()
        sys.exit(1)

async def cleanup():
    logger.info("Neko Bot is shutting down...")
    logger.info("Saving final guild settings...")
    save_settings() # Ensure settings are saved on shutdown
    
    # Cancel any pending asyncio tasks created by the bot
    # (excluding the current task, which is the cleanup itself)
    tasks_to_cancel = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if tasks_to_cancel:
        logger.info(f"Cancelling {len(tasks_to_cancel)} pending asyncio tasks...")
        for task in tasks_to_cancel:
            task.cancel()
        try:
            # Wait for tasks to acknowledge cancellation, with a timeout
            await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=5.0)
            logger.info("Pending tasks cancellation requested.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for tasks to cancel. Some tasks may not have cleaned up gracefully.")
        except Exception as e_task_cancel:
            logger.error(f"Error during asyncio task cancellation: {e_task_cancel}", exc_info=True)
            
    if bot and not bot.is_closed():
        logger.info("Closing Discord connection...")
        await bot.close()
        logger.info("Discord connection closed.")
    else:
        logger.info("Discord connection was already closed or bot object not initialized.")
        
    colorama.deinit() # Clean up colorama resources
    await asyncio.sleep(0.5) # Short pause to allow logs to flush
    logger.info("Neko Bot shutdown complete. Bye nya~")

if __name__ == "__main__":
    # Setup asyncio event loop
    # loop = asyncio.get_event_loop() # Deprecated in Python 3.10+ for this use case
    # asyncio.set_event_loop_policy(...) # If specific policy needed

    try:
        asyncio.run(main()) # Python 3.7+
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown...")
    except Exception as e_top_level: # Catch truly unexpected top-level errors
        logger.critical(f"FATAL UNHANDLED EXCEPTION in main execution block: {e_top_level}", exc_info=True)
        print(f"FATAL ERROR: {e_top_level}")
    finally:
        logger.info("Starting final cleanup process...")
        # Run cleanup in a new asyncio execution context if main loop is closed
        try:
            # Need to ensure loop is running or create one for cleanup if main one died
            cleanup_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cleanup_loop)
            cleanup_loop.run_until_complete(cleanup())
            cleanup_loop.close()
        except Exception as e_final_cleanup:
            logger.error(f"Error during final cleanup: {e_final_cleanup}", exc_info=True)
        finally:
            # Ensure colorama is deinitialized if it hasn't been.
            if colorama.initialised: colorama.deinit()
            logger.info("Event loop for cleanup closed. Bot fully shut down.")
            print("Neko Bot has shut down.")