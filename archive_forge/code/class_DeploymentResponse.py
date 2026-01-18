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
class DeploymentResponse(_DeploymentResponseBase):
    """A future-like object wrapping the result of a unary deployment handle call.

    From inside a deployment, a `DeploymentResponse` can be awaited to retrieve the
    output of the call without blocking the asyncio event loop.

    From outside a deployment, `.result()` can be used to retrieve the output in a
    blocking manner.

    Example:

    .. code-block:: python

        from ray import serve
        from ray.serve.handle import DeploymentHandle

        @serve.deployment
        class Downstream:
            def say_hi(self, message: str) -> str:
                return f"Hello {message}!"

        @serve.deployment
        class Caller:
            def __init__(self, handle: DeploymentHandle):
                self._downstream_handle = handle

        async def __call__(self, message: str) -> str:
            # Inside a deployment: `await` the result to enable concurrency.
            response = self._downstream_handle.say_hi.remote(message)
            return await response

        app = Caller.bind(Downstream.bind())
        handle: DeploymentHandle = serve.run(app)

        # Outside a deployment: call `.result()` to get output.
        response = handle.remote("world")
        assert response.result() == "Hello world!"

    A `DeploymentResponse` can be passed directly to another `DeploymentHandle` call
    without fetching the result to enable composing multiple deployments together.

    Example:

    .. code-block:: python

        from ray import serve
        from ray.serve.handle import DeploymentHandle

        @serve.deployment
        class Adder:
            def add(self, val: int) -> int:
                return val + 1

        @serve.deployment
        class Caller:
            def __init__(self, handle: DeploymentHandle):
                self._adder_handle = handle

        async def __call__(self, start: int) -> int:
            return await self._adder_handle.add.remote(
                # Pass the response directly to another handle call without awaiting.
                self._adder_handle.add.remote(start)
            )

        app = Caller.bind(Adder.bind())
        handle: DeploymentHandle = serve.run(app)
        assert handle.remote(0).result() == 2
    """

    def __await__(self):
        """Yields the final result of the deployment handle call."""
        obj_ref = (yield from asyncio.wrap_future(self._object_ref_future))
        result = (yield from obj_ref.__await__())
        return result

    def result(self, timeout_s: Optional[float]=None) -> Any:
        """Fetch the result of the handle call synchronously.

        This should *not* be used from within a deployment as it runs in an asyncio
        event loop. For model composition, `await` the response instead.

        If `timeout_s` is provided and the result is not available before the timeout,
        a `TimeoutError` is raised.
        """
        return ray.get(self._to_object_ref_sync(_record_telemetry=False), timeout=timeout_s)

    @DeveloperAPI
    async def _to_object_ref(self, _record_telemetry: bool=True) -> ray.ObjectRef:
        """Advanced API to convert the response to a Ray `ObjectRef`.

        This is used to pass the output of a `DeploymentHandle` call to a Ray task or
        actor method call.

        This method is `async def` because it will block until the handle call has been
        assigned to a replica actor. If there are many requests in flight and all
        replicas' queues are full, this may be a slow operation.
        """
        return await self._to_object_ref_or_gen(_record_telemetry=_record_telemetry)

    @DeveloperAPI
    def _to_object_ref_sync(self, _record_telemetry: bool=True, _allow_running_in_asyncio_loop: bool=False) -> ray.ObjectRef:
        """Advanced API to convert the response to a Ray `ObjectRef`.

        This is used to pass the output of a `DeploymentHandle` call to a Ray task or
        actor method call.

        This method is a *blocking* call because it will block until the handle call has
        been assigned to a replica actor. If there are many requests in flight and all
        replicas' queues are full, this may be a slow operation.

        From inside a deployment, `_to_object_ref` should be used instead to avoid
        blocking the asyncio event loop.
        """
        return self._to_object_ref_or_gen_sync(_record_telemetry=_record_telemetry, _allow_running_in_asyncio_loop=_allow_running_in_asyncio_loop)