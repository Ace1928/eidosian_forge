from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _do_set(self, level, node, i, val):
    if id(node) in self._dirty_nodes:
        ret = node
    else:
        ret = list(node)
        self._dirty_nodes[id(ret)] = True
    if level == 0:
        ret[i & BIT_MASK] = val
        self._cached_leafs[i >> SHIFT] = ret
    else:
        sub_index = i >> level & BIT_MASK
        ret[sub_index] = self._do_set(level - SHIFT, node[sub_index], i, val)
    return ret