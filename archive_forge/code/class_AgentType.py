import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
from typing import List, Tuple
from enum import Enum
class AgentType(Enum):
    AGENT = 'Agent'
    USER = 'User'