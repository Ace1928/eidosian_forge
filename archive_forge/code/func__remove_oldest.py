from collections import deque
from typing import Any, Callable, Deque, Dict
def _remove_oldest(self):
    """Remove the oldest entry."""
    key = self._queue.popleft()
    self._remove(key)