import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _node_key(self, node):
    """Get the key for a node whether it's a tuple or node."""
    if isinstance(node, tuple):
        node = StaticTuple.from_sequence(node)
    if isinstance(node, StaticTuple):
        return node
    else:
        return node._key