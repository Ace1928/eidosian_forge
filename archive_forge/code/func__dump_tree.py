import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _dump_tree(self, include_keys=False, encoding='utf-8'):
    """Return the tree in a string representation."""
    self._ensure_root()

    def decode(x):
        return x.decode(encoding)
    res = self._dump_tree_node(self._root_node, prefix=b'', indent='', decode=decode, include_keys=include_keys)
    res.append('')
    return '\n'.join(res)