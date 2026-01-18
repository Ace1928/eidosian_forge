from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _index_or_slice(index, stop):
    if stop is None:
        return index
    return slice(index, stop)