from __future__ import annotations
import ast
import asyncio
import inspect
import textwrap
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
from langchain_core.pydantic_v1 import BaseConfig, BaseModel
from langchain_core.pydantic_v1 import create_model as _create_model_base
from langchain_core.runnables.schema import StreamEvent
class ConfigurableFieldSingleOption(NamedTuple):
    """Field that can be configured by the user with a default value."""
    id: str
    options: Mapping[str, Any]
    default: str
    name: Optional[str] = None
    description: Optional[str] = None
    is_shared: bool = False

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), self.default))