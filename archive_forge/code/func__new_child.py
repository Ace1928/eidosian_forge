import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _new_child(self, search_key, klass):
    """Create a new child node of type klass."""
    child = klass()
    child.set_maximum_size(self._maximum_size)
    child._key_width = self._key_width
    child._search_key_func = self._search_key_func
    self._items[search_key] = child
    return child