import logging
import queue
import time
from typing import Any, Callable, List, Sequence
import uuid
class QueueCallbackWorker(object):
    """A helper that executes a callback for items sent in a queue.

    Calls a blocking ``get()`` on the ``queue`` until it encounters
    :attr:`STOP`.

    Args:
        queue:
            A Queue instance, appropriate for crossing the concurrency boundary
            implemented by ``executor``. Items will be popped off (with a blocking
            ``get()``) until :attr:`STOP` is encountered.
        callback:
            A callback that can process items pulled off of the queue. Multiple items
            will be passed to the callback in batches.
        max_items:
            The maximum amount of items that will be passed to the callback at a time.
        max_latency:
            The maximum amount of time in seconds to wait for additional items before
            executing the callback.
    """

    def __init__(self, queue: queue.Queue, callback: Callable[[Sequence[Any]], Any], max_items: int=100, max_latency: float=0):
        self.queue = queue
        self._callback = callback
        self.max_items = max_items
        self.max_latency = max_latency

    def __call__(self) -> None:
        continue_ = True
        while continue_:
            items = _get_many(self.queue, max_items=self.max_items, max_latency=self.max_latency)
            try:
                items = items[:items.index(STOP)]
                continue_ = False
            except ValueError:
                pass
            try:
                self._callback(items)
            except Exception as exc:
                _LOGGER.exception('Error in queue callback worker: %s', exc)
        _LOGGER.debug('Exiting the QueueCallbackWorker.')