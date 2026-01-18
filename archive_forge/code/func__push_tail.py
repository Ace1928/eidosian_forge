from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _push_tail(self, level, parent, tail_node):
    """
        if parent is leaf, insert node,
        else does it map to an existing child? ->
             node_to_insert = push node one more level
        else alloc new path

        return  node_to_insert placed in copy of parent
        """
    ret = list(parent)
    if level == SHIFT:
        ret.append(tail_node)
        return ret
    sub_index = self._count - 1 >> level & BIT_MASK
    if len(parent) > sub_index:
        ret[sub_index] = self._push_tail(level - SHIFT, parent[sub_index], tail_node)
        return ret
    ret.append(self._new_path(level - SHIFT, tail_node))
    return ret