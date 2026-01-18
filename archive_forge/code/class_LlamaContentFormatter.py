import json
import urllib.request
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator, validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
class LlamaContentFormatter(CustomOpenAIContentFormatter):
    """Deprecated: Kept for backwards compatibility

    Content formatter for Llama."""
    content_formatter: Any = None

    def __init__(self) -> None:
        super().__init__()
        warnings.warn('`LlamaContentFormatter` will be deprecated in the future. \n                Please use `CustomOpenAIContentFormatter` instead.  \n            ')