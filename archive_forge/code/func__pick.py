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
def _pick(self, input: Dict[str, Any]) -> Any:
    assert isinstance(input, dict), 'The input to RunnablePassthrough.assign() must be a dict.'
    if isinstance(self.keys, str):
        return input.get(self.keys)
    else:
        picked = {k: input.get(k) for k in self.keys if k in input}
        if picked:
            return AddableDict(picked)
        else:
            return None