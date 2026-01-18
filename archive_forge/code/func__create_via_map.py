import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
@classmethod
def _create_via_map(klass, store, initial_value, maximum_size=0, key_width=1, search_key_func=None):
    result = klass(store, None, search_key_func=search_key_func)
    result._root_node.set_maximum_size(maximum_size)
    result._root_node._key_width = key_width
    delta = []
    for key, value in initial_value.items():
        delta.append((None, key, value))
    root_key = result.apply_delta(delta)
    return root_key