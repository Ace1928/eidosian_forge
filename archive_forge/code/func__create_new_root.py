from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _create_new_root(self):
    new_shift = self._shift
    if self._count >> SHIFT > 1 << self._shift:
        new_root = [self._root, self._new_path(self._shift, self._tail)]
        new_shift += SHIFT
    else:
        new_root = self._push_tail(self._shift, self._root, self._tail)
    return (new_root, new_shift)