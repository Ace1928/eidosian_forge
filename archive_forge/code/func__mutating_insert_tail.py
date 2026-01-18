from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _mutating_insert_tail(self):
    self._root, self._shift = self._create_new_root()
    self._tail = []