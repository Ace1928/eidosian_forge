import logging
import queue
import time
from typing import Any, Callable, List, Sequence
import uuid
def _get_many(queue_: queue.Queue, max_items: int=None, max_latency: float=0) -> List[Any]:
    """Get multiple items from a Queue.

    Gets at least one (blocking) and at most ``max_items`` items
    (non-blocking) from a given Queue. Does not mark the items as done.

    Args:
        queue_: The Queue to get items from.
        max_items:
            The maximum number of items to get. If ``None``, then all available items
            in the queue are returned.
        max_latency:
            The maximum number of seconds to wait for more than one item from a queue.
            This number includes the time required to retrieve the first item.

    Returns:
        A sequence of items retrieved from the queue.
    """
    start = time.time()
    items = [queue_.get()]
    while max_items is None or len(items) < max_items:
        try:
            elapsed = time.time() - start
            timeout = max(0, max_latency - elapsed)
            items.append(queue_.get(timeout=timeout))
        except queue.Empty:
            break
    return items