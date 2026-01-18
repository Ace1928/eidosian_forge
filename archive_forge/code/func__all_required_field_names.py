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
@classmethod
def _all_required_field_names(cls) -> Set:
    """DEPRECATED: Kept for backwards compatibility.

        Use get_pydantic_field_names.
        """
    return get_pydantic_field_names(cls)