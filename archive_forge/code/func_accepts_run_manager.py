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
def accepts_run_manager(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a run_manager argument."""
    try:
        return signature(callable).parameters.get('run_manager') is not None
    except ValueError:
        return False