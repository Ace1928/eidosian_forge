from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _new_path(self, level, node):
    if level == 0:
        return node
    return [self._new_path(level - SHIFT, node)]