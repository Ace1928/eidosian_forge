from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
@staticmethod
def _node_for(pvector_like, i):
    if 0 <= i < pvector_like._count:
        if i >= pvector_like._tail_offset:
            return pvector_like._tail
        node = pvector_like._root
        for level in range(pvector_like._shift, 0, -SHIFT):
            node = node[i >> level & BIT_MASK]
        return node
    raise IndexError('Index out of range: %s' % (i,))