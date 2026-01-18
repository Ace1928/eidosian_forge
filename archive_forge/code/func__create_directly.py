import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
@classmethod
def _create_directly(klass, store, initial_value, maximum_size=0, key_width=1, search_key_func=None):
    node = LeafNode(search_key_func=search_key_func)
    node.set_maximum_size(maximum_size)
    node._key_width = key_width
    as_st = StaticTuple.from_sequence
    node._items = {as_st(key): val for key, val in initial_value.items()}
    node._raw_size = sum((node._key_value_len(key, value) for key, value in node._items.items()))
    node._len = len(node._items)
    node._compute_search_prefix()
    node._compute_serialised_prefix()
    if node._len > 1 and maximum_size and (node._current_size() > maximum_size):
        prefix, node_details = node._split(store)
        if len(node_details) == 1:
            raise AssertionError('Failed to split using node._split')
        node = InternalNode(prefix, search_key_func=search_key_func)
        node.set_maximum_size(maximum_size)
        node._key_width = key_width
        for split, subnode in node_details:
            node.add_node(split, subnode)
    keys = list(node.serialise(store))
    return keys[-1]