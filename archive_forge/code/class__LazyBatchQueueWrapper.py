import asyncio
import time
from dataclasses import dataclass
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import (
from ray._private.signature import extract_signature, flatten_args, recover_args
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.utils import extract_self_if_method_call
from ray.serve.exceptions import RayServeException
from ray.util.annotations import PublicAPI
class _LazyBatchQueueWrapper:
    """Stores a _BatchQueue and updates its settings.

    _BatchQueue cannot be pickled, you must construct it lazily
    at runtime inside a replica. This class initializes a queue only upon
    first access.
    """

    def __init__(self, max_batch_size: int=10, batch_wait_timeout_s: float=0.0, handle_batch_func: Optional[Callable]=None, batch_queue_cls: Type[_BatchQueue]=_BatchQueue):
        self._queue: Type[_BatchQueue] = None
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.handle_batch_func = handle_batch_func
        self.batch_queue_cls = batch_queue_cls

    @property
    def queue(self) -> Type[_BatchQueue]:
        """Returns _BatchQueue.

        Initializes queue when called for the first time.
        """
        if self._queue is None:
            self._queue = self.batch_queue_cls(self.max_batch_size, self.batch_wait_timeout_s, self.handle_batch_func)
        return self._queue

    def set_max_batch_size(self, new_max_batch_size: int) -> None:
        """Updates queue's max_batch_size."""
        self.max_batch_size = new_max_batch_size
        if self._queue is not None:
            self._queue.max_batch_size = new_max_batch_size

    def set_batch_wait_timeout_s(self, new_batch_wait_timeout_s: float) -> None:
        self.batch_wait_timeout_s = new_batch_wait_timeout_s
        if self._queue is not None:
            self._queue.batch_wait_timeout_s = new_batch_wait_timeout_s

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def get_batch_wait_timeout_s(self) -> float:
        return self.batch_wait_timeout_s