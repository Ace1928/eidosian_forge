import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
def _is_hex_string(data: str) -> bool:
    """Checks if a data is a valid hexadecimal string using a regular expression."""
    if not isinstance(data, str):
        return False
    pattern = '^[0-9a-fA-F]+$'
    return bool(re.match(pattern, data))