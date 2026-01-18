import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _check_remap(self, store):
    """Check if all keys contained by children fit in a single LeafNode.

        :param store: A store to use for reading more nodes
        :return: Either self, or a new LeafNode which should replace self.
        """
    new_leaf = LeafNode(search_key_func=self._search_key_func)
    new_leaf.set_maximum_size(self._maximum_size)
    new_leaf._key_width = self._key_width
    for node, _ in self._iter_nodes(store, batch_size=16):
        if isinstance(node, InternalNode):
            return self
        for key, value in node._items.items():
            if new_leaf._map_no_split(key, value):
                return self
    trace.mutter('remap generated a new LeafNode')
    return new_leaf