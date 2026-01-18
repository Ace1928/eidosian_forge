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
def _seq_output_schema(steps: List[Runnable[Any, Any]], config: Optional[RunnableConfig]) -> Type[BaseModel]:
    from langchain_core.runnables.passthrough import RunnableAssign, RunnablePick
    last = steps[-1]
    if len(steps) == 1:
        return last.get_input_schema(config)
    elif isinstance(last, RunnableAssign):
        mapper_output_schema = last.mapper.get_output_schema(config)
        prev_output_schema = _seq_output_schema(steps[:-1], config)
        if not prev_output_schema.__custom_root_type__:
            return create_model('RunnableSequenceOutput', **{**{k: (v.annotation, v.default) for k, v in prev_output_schema.__fields__.items()}, **{k: (v.annotation, v.default) for k, v in mapper_output_schema.__fields__.items()}})
    elif isinstance(last, RunnablePick):
        prev_output_schema = _seq_output_schema(steps[:-1], config)
        if not prev_output_schema.__custom_root_type__:
            if isinstance(last.keys, list):
                return create_model('RunnableSequenceOutput', **{k: (v.annotation, v.default) for k, v in prev_output_schema.__fields__.items() if k in last.keys})
            else:
                field = prev_output_schema.__fields__[last.keys]
                return create_model('RunnableSequenceOutput', __root__=(field.annotation, field.default))
    return last.get_output_schema(config)