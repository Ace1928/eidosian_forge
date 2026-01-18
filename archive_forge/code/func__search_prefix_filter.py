import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _search_prefix_filter(self, key):
    """Serialise key for use as a prefix filter in iteritems."""
    return self._search_key_func(key)[:self._node_width]