from __future__ import annotations
import inspect
import uuid
import warnings
from abc import abstractmethod
from functools import partial
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.callbacks.manager import (
from langchain_core.load.serializable import Serializable
from langchain_core.prompts import (
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
def _make_with_name(tool_name: str) -> Callable:

    def _make_tool(dec_func: Union[Callable, Runnable]) -> BaseTool:
        if isinstance(dec_func, Runnable):
            runnable = dec_func
            if runnable.input_schema.schema().get('type') != 'object':
                raise ValueError('Runnable must have an object schema.')

            async def ainvoke_wrapper(callbacks: Optional[Callbacks]=None, **kwargs: Any) -> Any:
                return await runnable.ainvoke(kwargs, {'callbacks': callbacks})

            def invoke_wrapper(callbacks: Optional[Callbacks]=None, **kwargs: Any) -> Any:
                return runnable.invoke(kwargs, {'callbacks': callbacks})
            coroutine = ainvoke_wrapper
            func = invoke_wrapper
            schema: Optional[Type[BaseModel]] = runnable.input_schema
            description = repr(runnable)
        elif inspect.iscoroutinefunction(dec_func):
            coroutine = dec_func
            func = None
            schema = args_schema
            description = None
        else:
            coroutine = None
            func = dec_func
            schema = args_schema
            description = None
        if infer_schema or args_schema is not None:
            return StructuredTool.from_function(func, coroutine, name=tool_name, description=description, return_direct=return_direct, args_schema=schema, infer_schema=infer_schema)
        if func.__doc__ is None:
            raise ValueError('Function must have a docstring if description not provided and infer_schema is False.')
        return Tool(name=tool_name, func=func, description=f'{tool_name} tool', return_direct=return_direct, coroutine=coroutine)
    return _make_tool