import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _are_search_keys_identical(self):
    """Check to see if the search keys for all entries are the same.

        When using a hash as the search_key it is possible for non-identical
        keys to collide. If that happens enough, we may try overflow a
        LeafNode, but as all are collisions, we must not split.
        """
    common_search_key = None
    for key in self._items:
        search_key = self._search_key(key)
        if common_search_key is None:
            common_search_key = search_key
        elif search_key != common_search_key:
            return False
    return True