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
class RunnableParallel(RunnableSerializable[Input, Dict[str, Any]]):
    """Runnable that runs a mapping of Runnables in parallel, and returns a mapping
    of their outputs.

    RunnableParallel is one of the two main composition primitives for the LCEL,
    alongside RunnableSequence. It invokes Runnables concurrently, providing the same
    input to each.

    A RunnableParallel can be instantiated directly or by using a dict literal within a
    sequence.

    Here is a simple example that uses functions to illustrate the use of
    RunnableParallel:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            def mul_two(x: int) -> int:
                return x * 2

            def mul_three(x: int) -> int:
                return x * 3

            runnable_1 = RunnableLambda(add_one)
            runnable_2 = RunnableLambda(mul_two)
            runnable_3 = RunnableLambda(mul_three)

            sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
                "mul_two": runnable_2,
                "mul_three": runnable_3,
            }
            # Or equivalently:
            # sequence = runnable_1 | RunnableParallel(
            #     {"mul_two": runnable_2, "mul_three": runnable_3}
            # )
            # Also equivalently:
            # sequence = runnable_1 | RunnableParallel(
            #     mul_two=runnable_2,
            #     mul_three=runnable_3,
            # )

            sequence.invoke(1)
            await sequence.ainvoke(1)

            sequence.batch([1, 2, 3])
            await sequence.abatch([1, 2, 3])

    RunnableParallel makes it easy to run Runnables in parallel. In the below example,
    we simultaneously stream output from two different Runnables:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableParallel
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI()
            joke_chain = (
                ChatPromptTemplate.from_template("tell me a joke about {topic}")
                | model
            )
            poem_chain = (
                ChatPromptTemplate.from_template("write a 2-line poem about {topic}")
                | model
            )

            runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

            # Display stream
            output = {key: "" for key, _ in runnable.output_schema()}
            for chunk in runnable.stream({"topic": "bear"}):
                for key in chunk:
                    output[key] = output[key] + chunk[key].content
                print(output)  # noqa: T201
    """
    steps__: Mapping[str, Runnable[Input, Any]]

    def __init__(self, steps__: Optional[Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any], Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]]]]]=None, **kwargs: Union[Runnable[Input, Any], Callable[[Input], Any], Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]]]) -> None:
        merged = {**steps__} if steps__ is not None else {}
        merged.update(kwargs)
        super().__init__(steps__={key: coerce_to_runnable(r) for key, r in merged.items()})

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    class Config:
        arbitrary_types_allowed = True

    def get_name(self, suffix: Optional[str]=None, *, name: Optional[str]=None) -> str:
        name = name or self.name or f'RunnableParallel<{','.join(self.steps__.keys())}>'
        return super().get_name(suffix, name=name)

    @property
    def InputType(self) -> Any:
        for step in self.steps__.values():
            if step.InputType:
                return step.InputType
        return Any

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        if all((s.get_input_schema(config).schema().get('type', 'object') == 'object' for s in self.steps__.values())):
            return create_model(self.get_name('Input'), **{k: (v.annotation, v.default) for step in self.steps__.values() for k, v in step.get_input_schema(config).__fields__.items() if k != '__root__'})
        return super().get_input_schema(config)

    def get_output_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        return create_model(self.get_name('Output'), **{k: (v.OutputType, None) for k, v in self.steps__.items()})

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs((spec for step in self.steps__.values() for spec in step.config_specs))

    def get_graph(self, config: Optional[RunnableConfig]=None) -> Graph:
        from langchain_core.runnables.graph import Graph
        graph = Graph()
        input_node = graph.add_node(self.get_input_schema(config))
        output_node = graph.add_node(self.get_output_schema(config))
        for step in self.steps__.values():
            step_graph = step.get_graph()
            step_graph.trim_first_node()
            step_graph.trim_last_node()
            if not step_graph:
                graph.add_edge(input_node, output_node)
            else:
                graph.extend(step_graph)
                step_first_node = step_graph.first_node()
                if not step_first_node:
                    raise ValueError(f'Runnable {step} has no first node')
                step_last_node = step_graph.last_node()
                if not step_last_node:
                    raise ValueError(f'Runnable {step} has no last node')
                graph.add_edge(input_node, step_first_node)
                graph.add_edge(step_last_node, output_node)
        return graph

    def __repr__(self) -> str:
        map_for_repr = ',\n  '.join((f'{k}: {indent_lines_after_first(repr(v), '  ' + k + ': ')}' for k, v in self.steps__.items()))
        return '{\n  ' + map_for_repr + '\n}'

    def invoke(self, input: Input, config: Optional[RunnableConfig]=None) -> Dict[str, Any]:
        from langchain_core.callbacks.manager import CallbackManager
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(inheritable_callbacks=config.get('callbacks'), local_callbacks=None, verbose=False, inheritable_tags=config.get('tags'), local_tags=None, inheritable_metadata=config.get('metadata'), local_metadata=None)
        run_manager = callback_manager.on_chain_start(dumpd(self), input, name=config.get('run_name') or self.get_name(), run_id=config.pop('run_id', None))
        try:
            steps = dict(self.steps__)
            with get_executor_for_config(config) as executor:
                futures = [executor.submit(step.invoke, input, patch_config(config, callbacks=run_manager.get_child(f'map:key:{key}'))) for key, step in steps.items()]
                output = {key: future.result() for key, future in zip(steps, futures)}
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Dict[str, Any]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(dumpd(self), input, name=config.get('run_name') or self.get_name(), run_id=config.pop('run_id', None))
        try:
            steps = dict(self.steps__)
            results = await asyncio.gather(*(step.ainvoke(input, patch_config(config, callbacks=run_manager.get_child(f'map:key:{key}'))) for key, step in steps.items()))
            output = {key: value for key, value in zip(steps, results)}
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output

    def _transform(self, input: Iterator[Input], run_manager: CallbackManagerForChainRun, config: RunnableConfig) -> Iterator[AddableDict]:
        steps = dict(self.steps__)
        input_copies = list(safetee(input, len(steps), lock=threading.Lock()))
        with get_executor_for_config(config) as executor:
            named_generators = [(name, step.transform(input_copies.pop(), patch_config(config, callbacks=run_manager.get_child(f'map:key:{name}')))) for name, step in steps.items()]
            futures = {executor.submit(next, generator): (step_name, generator) for step_name, generator in named_generators}
            while futures:
                completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    step_name, generator = futures.pop(future)
                    try:
                        chunk = AddableDict({step_name: future.result()})
                        yield chunk
                        futures[executor.submit(next, generator)] = (step_name, generator)
                    except StopIteration:
                        pass

    def transform(self, input: Iterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(input, self._transform, config, **kwargs)

    def stream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[Dict[str, Any]]:
        yield from self.transform(iter([input]), config)

    async def _atransform(self, input: AsyncIterator[Input], run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig) -> AsyncIterator[AddableDict]:
        steps = dict(self.steps__)
        input_copies = list(atee(input, len(steps), lock=asyncio.Lock()))
        named_generators = [(name, step.atransform(input_copies.pop(), patch_config(config, callbacks=run_manager.get_child(f'map:key:{name}')))) for name, step in steps.items()]

        async def get_next_chunk(generator: AsyncIterator) -> Optional[Output]:
            return await py_anext(generator)
        tasks = {asyncio.create_task(get_next_chunk(generator)): (step_name, generator) for step_name, generator in named_generators}
        while tasks:
            completed_tasks, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in completed_tasks:
                step_name, generator = tasks.pop(task)
                try:
                    chunk = AddableDict({step_name: task.result()})
                    yield chunk
                    new_task = asyncio.create_task(get_next_chunk(generator))
                    tasks[new_task] = (step_name, generator)
                except StopAsyncIteration:
                    pass

    async def atransform(self, input: AsyncIterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(input, self._atransform, config, **kwargs):
            yield chunk

    async def astream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[Dict[str, Any]]:

        async def input_aiter() -> AsyncIterator[Input]:
            yield input
        async for chunk in self.atransform(input_aiter(), config):
            yield chunk