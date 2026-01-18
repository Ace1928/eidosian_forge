import asyncio
import logging
import time
from abc import ABC, abstractmethod
from asyncio.tasks import FIRST_COMPLETED
from typing import Any, Callable, Optional, Union
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import calculate_remaining_timeout
from ray.serve.handle import DeploymentResponse, DeploymentResponseGenerator
class _ProxyResponseGeneratorBase(ABC):

    def __init__(self, *, timeout_s: Optional[float]=None, disconnected_task: Optional[asyncio.Task]=None, result_callback: Optional[Callable[[Any], Any]]=None):
        """Implements a generator wrapping a deployment response.

        Args:
            - timeout_s: an end-to-end timeout for the request. If this expires and the
              response is not completed, the request will be cancelled. If `None`,
              there's no timeout.
            - disconnected_task: a task whose completion signals that the client has
              disconnected. When this happens, the request will be cancelled. If `None`,
              disconnects will not be detected.
            - result_callback: will be called on each result before it's returned. If
              `None`, the unmodified result is returned.
        """
        self._timeout_s = timeout_s
        self._start_time_s = time.time()
        self._disconnected_task = disconnected_task
        self._result_callback = result_callback

    def __aiter__(self):
        return self

    @abstractmethod
    async def __anext__(self):
        """Return the next message in the stream.

        Raises:
            - TimeoutError on timeout.
            - asyncio.CancelledError on disconnect.
            - StopAsyncIteration when the stream is completed.
        """
        pass

    def stop_checking_for_disconnect(self):
        """Once this is called, the disconnected_task will be ignored."""
        self._disconnected_task = None