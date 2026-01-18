import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class _Iter:
    __slots__ = ('_size', '_iter')

    def __init__(self, size, iterator):
        self._size = size
        self._iter = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __length_hint__(self):
        return self._size