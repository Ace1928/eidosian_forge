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
class RunnableBinding(RunnableBindingBase[Input, Output]):
    """Wrap a Runnable with additional functionality.

    A RunnableBinding can be thought of as a "runnable decorator" that
    preserves the essential features of Runnable; i.e., batching, streaming,
    and async support, while adding additional functionality.

    Any class that inherits from Runnable can be bound to a `RunnableBinding`.
    Runnables expose a standard set of methods for creating `RunnableBindings`
    or sub-classes of `RunnableBindings` (e.g., `RunnableRetry`,
    `RunnableWithFallbacks`) that add additional functionality.

    These methods include:
    - `bind`: Bind kwargs to pass to the underlying runnable when running it.
    - `with_config`: Bind config to pass to the underlying runnable when running it.
    - `with_listeners`:  Bind lifecycle listeners to the underlying runnable.
    - `with_types`: Override the input and output types of the underlying runnable.
    - `with_retry`: Bind a retry policy to the underlying runnable.
    - `with_fallbacks`: Bind a fallback policy to the underlying runnable.

    Example:

    `bind`: Bind kwargs to pass to the underlying runnable when running it.

        .. code-block:: python

            # Create a runnable binding that invokes the ChatModel with the
            # additional kwarg `stop=['-']` when running it.
            from langchain_community.chat_models import ChatOpenAI
            model = ChatOpenAI()
            model.invoke('Say "Parrot-MAGIC"', stop=['-']) # Should return `Parrot`
            # Using it the easy way via `bind` method which returns a new
            # RunnableBinding
            runnable_binding = model.bind(stop=['-'])
            runnable_binding.invoke('Say "Parrot-MAGIC"') # Should return `Parrot`

        Can also be done by instantiating a RunnableBinding directly (not recommended):

        .. code-block:: python

            from langchain_core.runnables import RunnableBinding
            runnable_binding = RunnableBinding(
                bound=model,
                kwargs={'stop': ['-']} # <-- Note the additional kwargs
            )
            runnable_binding.invoke('Say "Parrot-MAGIC"') # Should return `Parrot`
    """

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """Bind additional kwargs to a Runnable, returning a new Runnable.

        Args:
            **kwargs: The kwargs to bind to the Runnable.

        Returns:
            A new Runnable with the same type and config as the original,
            but with the additional kwargs bound.
        """
        return self.__class__(bound=self.bound, config=self.config, kwargs={**self.kwargs, **kwargs}, custom_input_type=self.custom_input_type, custom_output_type=self.custom_output_type)

    def with_config(self, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(bound=self.bound, kwargs=self.kwargs, config=cast(RunnableConfig, {**self.config, **(config or {}), **kwargs}), custom_input_type=self.custom_input_type, custom_output_type=self.custom_output_type)

    def with_listeners(self, *, on_start: Optional[Listener]=None, on_end: Optional[Listener]=None, on_error: Optional[Listener]=None) -> Runnable[Input, Output]:
        """Bind lifecycle listeners to a Runnable, returning a new Runnable.

        Args:
            on_start: Called before the runnable starts running, with the Run object.
            on_end: Called after the runnable finishes running, with the Run object.
            on_error: Called if the runnable throws an error, with the Run object.

        Returns:
            The Run object contains information about the run, including its id,
            type, input, output, error, start_time, end_time, and any tags or metadata
            added to the run.
        """
        from langchain_core.tracers.root_listeners import RootListenersTracer
        return self.__class__(bound=self.bound, kwargs=self.kwargs, config=self.config, config_factories=[lambda config: {'callbacks': [RootListenersTracer(config=config, on_start=on_start, on_end=on_end, on_error=on_error)]}], custom_input_type=self.custom_input_type, custom_output_type=self.custom_output_type)

    def with_types(self, input_type: Optional[Union[Type[Input], BaseModel]]=None, output_type: Optional[Union[Type[Output], BaseModel]]=None) -> Runnable[Input, Output]:
        return self.__class__(bound=self.bound, kwargs=self.kwargs, config=self.config, custom_input_type=input_type if input_type is not None else self.custom_input_type, custom_output_type=output_type if output_type is not None else self.custom_output_type)

    def with_retry(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(bound=self.bound.with_retry(**kwargs), kwargs=self.kwargs, config=self.config)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.bound, name)
        if callable(attr) and (config_param := inspect.signature(attr).parameters.get('config')):
            if config_param.kind == inspect.Parameter.KEYWORD_ONLY:

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    return attr(*args, config=merge_configs(self.config, kwargs.pop('config', None)), **kwargs)
                return wrapper
            elif config_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                idx = list(inspect.signature(attr).parameters).index('config')

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    if len(args) >= idx + 1:
                        argsl = list(args)
                        argsl[idx] = merge_configs(self.config, argsl[idx])
                        return attr(*argsl, **kwargs)
                    else:
                        return attr(*args, config=merge_configs(self.config, kwargs.pop('config', None)), **kwargs)
                return wrapper
        return attr