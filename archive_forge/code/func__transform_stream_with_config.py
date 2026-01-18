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
def _transform_stream_with_config(self, input: Iterator[Input], transformer: Union[Callable[[Iterator[Input]], Iterator[Output]], Callable[[Iterator[Input], CallbackManagerForChainRun], Iterator[Output]], Callable[[Iterator[Input], CallbackManagerForChainRun, RunnableConfig], Iterator[Output]]], config: Optional[RunnableConfig], run_type: Optional[str]=None, **kwargs: Optional[Any]) -> Iterator[Output]:
    """Helper method to transform an Iterator of Input values into an Iterator of
        Output values, with callbacks.
        Use this to implement `stream()` or `transform()` in Runnable subclasses."""
    input_for_tracing, input_for_transform = tee(input, 2)
    final_input: Optional[Input] = next(input_for_tracing, None)
    final_input_supported = True
    final_output: Optional[Output] = None
    final_output_supported = True
    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
    run_manager = callback_manager.on_chain_start(dumpd(self), {'input': ''}, run_type=run_type, name=config.get('run_name') or self.get_name(), run_id=config.pop('run_id', None))
    try:
        child_config = patch_config(config, callbacks=run_manager.get_child())
        if accepts_config(transformer):
            kwargs['config'] = child_config
        if accepts_run_manager(transformer):
            kwargs['run_manager'] = run_manager
        context = copy_context()
        context.run(var_child_runnable_config.set, child_config)
        iterator = context.run(transformer, input_for_transform, **kwargs)
        try:
            while True:
                chunk: Output = context.run(next, iterator)
                yield chunk
                if final_output_supported:
                    if final_output is None:
                        final_output = chunk
                    else:
                        try:
                            final_output = final_output + chunk
                        except TypeError:
                            final_output = chunk
                            final_output_supported = False
                else:
                    final_output = chunk
        except StopIteration:
            pass
        for ichunk in input_for_tracing:
            if final_input_supported:
                if final_input is None:
                    final_input = ichunk
                else:
                    try:
                        final_input = final_input + ichunk
                    except TypeError:
                        final_input = ichunk
                        final_input_supported = False
            else:
                final_input = ichunk
    except BaseException as e:
        run_manager.on_chain_error(e, inputs=final_input)
        raise
    else:
        run_manager.on_chain_end(final_output, inputs=final_input)