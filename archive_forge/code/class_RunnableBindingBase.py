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
class RunnableBindingBase(RunnableSerializable[Input, Output]):
    """Runnable that delegates calls to another Runnable with a set of kwargs.

    Use only if creating a new RunnableBinding subclass with different __init__ args.

    See documentation for RunnableBinding for more details.
    """
    bound: Runnable[Input, Output]
    'The underlying runnable that this runnable delegates to.'
    kwargs: Mapping[str, Any] = Field(default_factory=dict)
    'kwargs to pass to the underlying runnable when running.\n\n    For example, when the runnable binding is invoked the underlying\n    runnable will be invoked with the same input but with these additional\n    kwargs.\n    '
    config: RunnableConfig = Field(default_factory=dict)
    'The config to bind to the underlying runnable.'
    config_factories: List[Callable[[RunnableConfig], RunnableConfig]] = Field(default_factory=list)
    'The config factories to bind to the underlying runnable.'
    custom_input_type: Optional[Any] = None
    'Override the input type of the underlying runnable with a custom type.\n\n    The type can be a pydantic model, or a type annotation (e.g., `List[str]`).\n    '
    custom_output_type: Optional[Any] = None
    'Override the output type of the underlying runnable with a custom type.\n\n    The type can be a pydantic model, or a type annotation (e.g., `List[str]`).\n    '

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *, bound: Runnable[Input, Output], kwargs: Optional[Mapping[str, Any]]=None, config: Optional[RunnableConfig]=None, config_factories: Optional[List[Callable[[RunnableConfig], RunnableConfig]]]=None, custom_input_type: Optional[Union[Type[Input], BaseModel]]=None, custom_output_type: Optional[Union[Type[Output], BaseModel]]=None, **other_kwargs: Any) -> None:
        """Create a RunnableBinding from a runnable and kwargs.

        Args:
            bound: The underlying runnable that this runnable delegates calls to.
            kwargs: optional kwargs to pass to the underlying runnable, when running
                    the underlying runnable (e.g., via `invoke`, `batch`,
                    `transform`, or `stream` or async variants)
            config: config_factories:
            config_factories: optional list of config factories to apply to the
            custom_input_type: Specify to override the input type of the underlying
                               runnable with a custom type.
            custom_output_type: Specify to override the output type of the underlying
                runnable with a custom type.
            **other_kwargs: Unpacked into the base class.
        """
        super().__init__(bound=bound, kwargs=kwargs or {}, config=config or {}, config_factories=config_factories or [], custom_input_type=custom_input_type, custom_output_type=custom_output_type, **other_kwargs)
        self.config = config or {}

    def get_name(self, suffix: Optional[str]=None, *, name: Optional[str]=None) -> str:
        return self.bound.get_name(suffix, name=name)

    @property
    def InputType(self) -> Type[Input]:
        return cast(Type[Input], self.custom_input_type) if self.custom_input_type is not None else self.bound.InputType

    @property
    def OutputType(self) -> Type[Output]:
        return cast(Type[Output], self.custom_output_type) if self.custom_output_type is not None else self.bound.OutputType

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        if self.custom_input_type is not None:
            return super().get_input_schema(config)
        return self.bound.get_input_schema(merge_configs(self.config, config))

    def get_output_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        if self.custom_output_type is not None:
            return super().get_output_schema(config)
        return self.bound.get_output_schema(merge_configs(self.config, config))

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.bound.config_specs

    def get_graph(self, config: Optional[RunnableConfig]=None) -> Graph:
        return self.bound.get_graph(config)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = merge_configs(self.config, *configs)
        return merge_configs(config, *(f(config) for f in self.config_factories))

    def invoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Output:
        return self.bound.invoke(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Output:
        return await self.bound.ainvoke(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    def batch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> List[Output]:
        if isinstance(config, list):
            configs = cast(List[RunnableConfig], [self._merge_configs(conf) for conf in config])
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        return self.bound.batch(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs})

    async def abatch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> List[Output]:
        if isinstance(config, list):
            configs = cast(List[RunnableConfig], [self._merge_configs(conf) for conf in config])
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        return await self.bound.abatch(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs})

    @overload
    def batch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: Literal[False]=False, **kwargs: Any) -> Iterator[Tuple[int, Output]]:
        ...

    @overload
    def batch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: Literal[True], **kwargs: Any) -> Iterator[Tuple[int, Union[Output, Exception]]]:
        ...

    def batch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> Iterator[Tuple[int, Union[Output, Exception]]]:
        if isinstance(config, list):
            configs = cast(List[RunnableConfig], [self._merge_configs(conf) for conf in config])
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        if return_exceptions:
            yield from self.bound.batch_as_completed(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs})
        else:
            yield from self.bound.batch_as_completed(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs})

    @overload
    def abatch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: Literal[False]=False, **kwargs: Optional[Any]) -> AsyncIterator[Tuple[int, Output]]:
        ...

    @overload
    def abatch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: Literal[True], **kwargs: Optional[Any]) -> AsyncIterator[Tuple[int, Union[Output, Exception]]]:
        ...

    async def abatch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> AsyncIterator[Tuple[int, Union[Output, Exception]]]:
        if isinstance(config, list):
            configs = cast(List[RunnableConfig], [self._merge_configs(conf) for conf in config])
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        if return_exceptions:
            async for item in self.bound.abatch_as_completed(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs}):
                yield item
        else:
            async for item in self.bound.abatch_as_completed(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs}):
                yield item

    def stream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[Output]:
        yield from self.bound.stream(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    async def astream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[Output]:
        async for item in self.bound.astream(input, self._merge_configs(config), **{**self.kwargs, **kwargs}):
            yield item

    async def astream_events(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[StreamEvent]:
        async for item in self.bound.astream_events(input, self._merge_configs(config), **{**self.kwargs, **kwargs}):
            yield item

    def transform(self, input: Iterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Output]:
        yield from self.bound.transform(input, self._merge_configs(config), **{**self.kwargs, **kwargs})

    async def atransform(self, input: AsyncIterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Output]:
        async for item in self.bound.atransform(input, self._merge_configs(config), **{**self.kwargs, **kwargs}):
            yield item