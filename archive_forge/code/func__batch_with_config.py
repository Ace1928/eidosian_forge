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
def _batch_with_config(self, func: Union[Callable[[List[Input]], List[Union[Exception, Output]]], Callable[[List[Input], List[CallbackManagerForChainRun]], List[Union[Exception, Output]]], Callable[[List[Input], List[CallbackManagerForChainRun], List[RunnableConfig]], List[Union[Exception, Output]]]], input: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, run_type: Optional[str]=None, **kwargs: Optional[Any]) -> List[Output]:
    """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
    if not input:
        return []
    configs = get_config_list(config, len(input))
    callback_managers = [get_callback_manager_for_config(c) for c in configs]
    run_managers = [callback_manager.on_chain_start(dumpd(self), input, run_type=run_type, name=config.get('run_name') or self.get_name(), run_id=config.pop('run_id', None)) for callback_manager, input, config in zip(callback_managers, input, configs)]
    try:
        if accepts_config(func):
            kwargs['config'] = [patch_config(c, callbacks=rm.get_child()) for c, rm in zip(configs, run_managers)]
        if accepts_run_manager(func):
            kwargs['run_manager'] = run_managers
        output = func(input, **kwargs)
    except BaseException as e:
        for run_manager in run_managers:
            run_manager.on_chain_error(e)
        if return_exceptions:
            return cast(List[Output], [e for _ in input])
        else:
            raise
    else:
        first_exception: Optional[Exception] = None
        for run_manager, out in zip(run_managers, output):
            if isinstance(out, Exception):
                first_exception = first_exception or out
                run_manager.on_chain_error(out)
            else:
                run_manager.on_chain_end(out)
        if return_exceptions or first_exception is None:
            return cast(List[Output], output)
        else:
            raise first_exception