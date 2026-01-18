from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
@staticmethod
def _wrap_async(afunc: Optional[Callable[..., Awaitable[Output]]]) -> Optional[Callable[[Input, RunnableConfig], Awaitable[Output]]]:
    """Wrap an async function to make it synchronous."""
    if afunc is None:
        return None
    if not is_traceable_function(afunc):
        raise ValueError(f'RunnableTraceable expects a function wrapped by the LangSmith @traceable decorator. Got {afunc}')
    afunc_ = cast(Callable[..., Awaitable[Output]], afunc)

    async def awrap_traceable(inputs: dict, config: RunnableConfig) -> Any:
        run_tree = RunnableTraceable._configure_run_tree(config.get('callbacks'))
        return await afunc_(**inputs, langsmith_extra={'run_tree': run_tree})
    return cast(Callable[[Input, RunnableConfig], Awaitable[Output]], awrap_traceable)