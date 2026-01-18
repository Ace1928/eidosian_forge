from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
from typing_extensions import TypeAlias
from langchain_core._api import beta, deprecated
from langchain_core.messages import (
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_core.utils import get_pydantic_field_names
def _get_token_ids_default_method(text: str) -> List[int]:
    """Encode the text into token IDs."""
    tokenizer = get_tokenizer()
    return tokenizer.encode(text)