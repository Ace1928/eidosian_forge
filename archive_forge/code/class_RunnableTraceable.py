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
class RunnableTraceable(RunnableLambda):
    """Converts a @traceable decorated function to a Runnable.

        This helps maintain a consistent LangSmith tracing context.
        """

    def __init__(self, func: Callable, afunc: Optional[Callable[..., Awaitable[Output]]]=None) -> None:
        wrapped: Optional[Callable[[Input], Output]] = None
        awrapped = self._wrap_async(afunc)
        if is_async(func):
            if awrapped is not None:
                raise TypeError('Func was provided as a coroutine function, but afunc was also provided. If providing both, func should be a regular function to avoid ambiguity.')
            wrapped = cast(Callable[[Input], Output], self._wrap_async(func))
        elif is_traceable_function(func):
            wrapped = cast(Callable[[Input], Output], self._wrap_sync(func))
        if wrapped is None:
            raise ValueError(f'{self.__class__.__name__} expects a function wrapped by the LangSmith @traceable decorator. Got {func}')
        super().__init__(wrapped, cast(Optional[Callable[[Input], Awaitable[Output]]], awrapped))

    @staticmethod
    def _configure_run_tree(callback_manager: Any) -> Optional[run_trees.RunTree]:
        run_tree: Optional[run_trees.RunTree] = None
        if isinstance(callback_manager, (CallbackManager, AsyncCallbackManager)):
            lc_tracers = [handler for handler in callback_manager.handlers if isinstance(handler, LangChainTracer)]
            if lc_tracers:
                lc_tracer = lc_tracers[0]
                run_tree = run_trees.RunTree(id=callback_manager.parent_run_id, session_name=lc_tracer.project_name, name='Wrapping', run_type='chain', inputs={}, tags=callback_manager.tags, extra={'metadata': callback_manager.metadata})
        return run_tree

    @staticmethod
    def _wrap_sync(func: Callable[..., Output]) -> Callable[[Input, RunnableConfig], Output]:
        """Wrap a synchronous function to make it asynchronous."""

        def wrap_traceable(inputs: dict, config: RunnableConfig) -> Any:
            run_tree = RunnableTraceable._configure_run_tree(config.get('callbacks'))
            return func(**inputs, langsmith_extra={'run_tree': run_tree})
        return cast(Callable[[Input, RunnableConfig], Output], wrap_traceable)

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