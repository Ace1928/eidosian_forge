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
def _call_with_config(self, func: Union[Callable[[Input], Output], Callable[[Input, CallbackManagerForChainRun], Output], Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]], input: Input, config: Optional[RunnableConfig], run_type: Optional[str]=None, **kwargs: Optional[Any]) -> Output:
    """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
    run_manager = callback_manager.on_chain_start(dumpd(self), input, run_type=run_type, name=config.get('run_name') or self.get_name(), run_id=config.pop('run_id', None))
    try:
        child_config = patch_config(config, callbacks=run_manager.get_child())
        context = copy_context()
        context.run(var_child_runnable_config.set, child_config)
        output = cast(Output, context.run(call_func_with_variable_args, func, input, config, run_manager, **kwargs))
    except BaseException as e:
        run_manager.on_chain_error(e)
        raise
    else:
        run_manager.on_chain_end(output)
        return output