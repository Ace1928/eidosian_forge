from __future__ import annotations
import asyncio
import inspect
import threading
from typing import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import (
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
@property
def InputType(self) -> Any:
    return self.input_type or Any