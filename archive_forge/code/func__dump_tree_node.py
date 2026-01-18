import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _dump_tree_node(self, node, prefix, indent, decode, include_keys=True):
    """For this node and all children, generate a string representation."""
    result = []
    if not include_keys:
        key_str = ''
    else:
        node_key = node.key()
        if node_key is not None:
            key_str = ' {}'.format(decode(node_key[0]))
        else:
            key_str = ' None'
    result.append('{}{!r} {}{}'.format(indent, decode(prefix), node.__class__.__name__, key_str))
    if isinstance(node, InternalNode):
        list(node._iter_nodes(self._store))
        for prefix, sub in sorted(node._items.items()):
            result.extend(self._dump_tree_node(sub, prefix, indent + '  ', decode=decode, include_keys=include_keys))
    else:
        for key, value in sorted(node._items.items()):
            result.append('      {!r} {!r}'.format(tuple([decode(ke) for ke in key]), decode(value)))
    return result