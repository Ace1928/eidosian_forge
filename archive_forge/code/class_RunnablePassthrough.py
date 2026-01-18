from __future__ import annotations
import asyncio
import inspect
import threading
from typing import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import (
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """Runnable to passthrough inputs unchanged or with additional keys.

    This runnable behaves almost like the identity function, except that it
    can be configured to add additional keys to the output, if the input is a
    dict.

    The examples below demonstrate this Runnable works using a few simple
    chains. The chains rely on simple lambdas to make the examples easy to execute
    and experiment with.

    Examples:

        .. code-block:: python

            from langchain_core.runnables import (
                RunnableLambda,
                RunnableParallel,
                RunnablePassthrough,
            )

            runnable = RunnableParallel(
                origin=RunnablePassthrough(),
                modified=lambda x: x+1
            )

            runnable.invoke(1) # {'origin': 1, 'modified': 2}


            def fake_llm(prompt: str) -> str: # Fake LLM for the example
                return "completion"

            chain = RunnableLambda(fake_llm) | {
                'original': RunnablePassthrough(), # Original LLM output
                'parsed': lambda text: text[::-1] # Parsing logic
            }

            chain.invoke('hello') # {'original': 'completion', 'parsed': 'noitelpmoc'}

    In some cases, it may be useful to pass the input through while adding some
    keys to the output. In this case, you can use the `assign` method:

        .. code-block:: python

            from langchain_core.runnables import RunnablePassthrough

            def fake_llm(prompt: str) -> str: # Fake LLM for the example
                return "completion"

            runnable = {
                'llm1':  fake_llm,
                'llm2':  fake_llm,
            } | RunnablePassthrough.assign(
                total_chars=lambda inputs: len(inputs['llm1'] + inputs['llm2'])
            )

            runnable.invoke('hello')
            # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
    """
    input_type: Optional[Type[Other]] = None
    func: Optional[Union[Callable[[Other], None], Callable[[Other, RunnableConfig], None]]] = None
    afunc: Optional[Union[Callable[[Other], Awaitable[None]], Callable[[Other, RunnableConfig], Awaitable[None]]]] = None

    def __repr_args__(self) -> Any:
        return []

    def __init__(self, func: Optional[Union[Union[Callable[[Other], None], Callable[[Other, RunnableConfig], None]], Union[Callable[[Other], Awaitable[None]], Callable[[Other, RunnableConfig], Awaitable[None]]]]]=None, afunc: Optional[Union[Callable[[Other], Awaitable[None]], Callable[[Other, RunnableConfig], Awaitable[None]]]]=None, *, input_type: Optional[Type[Other]]=None, **kwargs: Any) -> None:
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None
        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    @property
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    def assign(cls, **kwargs: Union[Runnable[Dict[str, Any], Any], Callable[[Dict[str, Any]], Any], Mapping[str, Union[Runnable[Dict[str, Any], Any], Callable[[Dict[str, Any]], Any]]]]) -> 'RunnableAssign':
        """Merge the Dict input with the output produced by the mapping argument.

        Args:
            mapping: A mapping from keys to runnables or callables.

        Returns:
            A runnable that merges the Dict input with the output produced by the
            mapping argument.
        """
        return RunnableAssign(RunnableParallel(kwargs))

    def invoke(self, input: Other, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Other:
        if self.func is not None:
            call_func_with_variable_args(self.func, input, ensure_config(config), **kwargs)
        return self._call_with_config(identity, input, config)

    async def ainvoke(self, input: Other, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Other:
        if self.afunc is not None:
            await acall_func_with_variable_args(self.afunc, input, ensure_config(config), **kwargs)
        elif self.func is not None:
            call_func_with_variable_args(self.func, input, ensure_config(config), **kwargs)
        return await self._acall_with_config(aidentity, input, config)

    def transform(self, input: Iterator[Other], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Other]:
        if self.func is None:
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final = None
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
                if final is None:
                    final = adapt_first_streaming_chunk(chunk)
                else:
                    final = final + chunk
            if final is not None:
                call_func_with_variable_args(self.func, final, ensure_config(config), **kwargs)

    async def atransform(self, input: AsyncIterator[Other], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Other]:
        if self.afunc is None and self.func is None:
            async for chunk in self._atransform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final = None
            async for chunk in self._atransform_stream_with_config(input, identity, config):
                yield chunk
                if final is None:
                    final = adapt_first_streaming_chunk(chunk)
                else:
                    final = final + chunk
            if final is not None:
                config = ensure_config(config)
                if self.afunc is not None:
                    await acall_func_with_variable_args(self.afunc, final, config, **kwargs)
                elif self.func is not None:
                    call_func_with_variable_args(self.func, final, config, **kwargs)

    def stream(self, input: Other, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Other]:
        return self.transform(iter([input]), config, **kwargs)

    async def astream(self, input: Other, config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Other]:

        async def input_aiter() -> AsyncIterator[Other]:
            yield input
        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk