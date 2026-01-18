import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class InvalidToolCall(TypedDict):
    """Allowance for errors made by LLM.

    Here we add an `error` key to surface errors made during generation
    (e.g., invalid JSON arguments.)
    """
    name: Optional[str]
    args: Optional[str]
    id: Optional[str]
    error: Optional[str]