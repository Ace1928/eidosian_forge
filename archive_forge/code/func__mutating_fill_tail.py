from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _mutating_fill_tail(self, offset, sequence):
    max_delta_len = BRANCH_FACTOR - len(self._tail)
    delta = sequence[offset:offset + max_delta_len]
    self._tail.extend(delta)
    delta_len = len(delta)
    self._count += delta_len
    return offset + delta_len