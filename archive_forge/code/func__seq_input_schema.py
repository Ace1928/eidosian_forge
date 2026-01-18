from __future__ import annotations
import asyncio
import collections
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from typing import (
from typing_extensions import Literal, get_args
from langchain_core._api import beta_decorator
from langchain_core.load.dump import dumpd
from langchain_core.load.serializable import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
def _seq_input_schema(steps: List[Runnable[Any, Any]], config: Optional[RunnableConfig]) -> Type[BaseModel]:
    from langchain_core.runnables.passthrough import RunnableAssign, RunnablePick
    first = steps[0]
    if len(steps) == 1:
        return first.get_input_schema(config)
    elif isinstance(first, RunnableAssign):
        next_input_schema = _seq_input_schema(steps[1:], config)
        if not next_input_schema.__custom_root_type__:
            return create_model('RunnableSequenceInput', **{k: (v.annotation, v.default) for k, v in next_input_schema.__fields__.items() if k not in first.mapper.steps__})
    elif isinstance(first, RunnablePick):
        return _seq_input_schema(steps[1:], config)
    return first.get_input_schema(config)