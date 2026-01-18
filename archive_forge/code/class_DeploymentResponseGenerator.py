import asyncio
import concurrent.futures
import threading
import warnings
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union
import ray
from ray import serve
from ray._raylet import GcsClient, ObjectRefGenerator
from ray.serve._private.common import DeploymentID, RequestProtocol
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.router import RequestMetadata, Router
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.util import metrics
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@PublicAPI(stability='beta')
class DeploymentResponseGenerator(_DeploymentResponseBase):
    """A future-like object wrapping the result of a streaming deployment handle call.

    This is returned when using `handle.options(stream=True)` and calling a generator
    deployment method.

    `DeploymentResponseGenerator` is both a synchronous and asynchronous iterator.

    When iterating over results from inside a deployment, `async for` should be used to
    avoid blocking the asyncio event loop.

    When iterating over results from outside a deployment, use a standard `for` loop.

    Example:

    .. code-block:: python

        from typing import AsyncGenerator, Generator

        from ray import serve
        from ray.serve.handle import DeploymentHandle

        @serve.deployment
        class Streamer:
            def generate_numbers(self, limit: int) -> Generator[int]:
                for i in range(limit):
                    yield i

        @serve.deployment
        class Caller:
            def __init__(self, handle: DeploymentHandle):
                # Set `stream=True` on the handle to enable streaming calls.
                self._streaming_handle = handle.options(stream=True)

        async def __call__(self, limit: int) -> AsyncIterator[int]:
            gen: DeploymentResponseGenerator = (
                self._streaming_handle.generate_numbers.remote(limit)
            )

            # Inside a deployment: use `async for` to enable concurrency.
            async for i in gen:
                yield i

        app = Caller.bind(Streamer.bind())
        handle: DeploymentHandle = serve.run(app)

        # Outside a deployment: use a standard `for` loop.
        gen: DeploymentResponseGenerator = handle.options(stream=True).remote(10)
        assert [i for i in gen] == list(range(10))

    A `DeploymentResponseGenerator` *cannot* currently be passed to another
    `DeploymentHandle` call.
    """

    def __init__(self, object_ref_future: concurrent.futures.Future):
        super().__init__(object_ref_future)
        self._obj_ref_gen: Optional[ObjectRefGenerator] = None

    def __await__(self):
        raise TypeError('`DeploymentResponseGenerator` cannot be awaited directly. Use `async for` or `_to_object_ref_gen` instead.')

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if self._obj_ref_gen is None:
            self._obj_ref_gen = await self._to_object_ref_gen(_record_telemetry=False)
        next_obj_ref = await self._obj_ref_gen.__anext__()
        return await next_obj_ref

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._obj_ref_gen is None:
            self._obj_ref_gen = self._to_object_ref_gen_sync(_record_telemetry=False)
        next_obj_ref = self._obj_ref_gen.__next__()
        return ray.get(next_obj_ref)

    @DeveloperAPI
    async def _to_object_ref_gen(self, _record_telemetry: bool=True) -> ObjectRefGenerator:
        """Advanced API to convert the generator to a Ray `ObjectRefGenerator`.

        This method is `async def` because it will block until the handle call has been
        assigned to a replica actor. If there are many requests in flight and all
        replicas' queues are full, this may be a slow operation.
        """
        return await self._to_object_ref_or_gen(_record_telemetry=_record_telemetry)

    @DeveloperAPI
    def _to_object_ref_gen_sync(self, _record_telemetry: bool=True, _allow_running_in_asyncio_loop: bool=False) -> ObjectRefGenerator:
        """Advanced API to convert the generator to a Ray `ObjectRefGenerator`.

        This method is a *blocking* call because it will block until the handle call has
        been assigned to a replica actor. If there are many requests in flight and all
        replicas' queues are full, this may be a slow operation.

        From inside a deployment, `_to_object_ref_gen` should be used instead to avoid
        blocking the asyncio event loop.
        """
        return self._to_object_ref_or_gen_sync(_record_telemetry=_record_telemetry, _allow_running_in_asyncio_loop=_allow_running_in_asyncio_loop)