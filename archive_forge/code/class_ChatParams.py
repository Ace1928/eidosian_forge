import logging
import warnings
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import BaseModel, Extra
class ChatParams(BaseModel, extra=Extra.allow):
    """Parameters for the `MLflow AI Gateway` LLM."""
    temperature: float = 0.0
    candidate_count: int = 1
    'The number of candidates to return.'
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None