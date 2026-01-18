import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class _Impl:
    __slots__ = ('_items', '_version')

    def __init__(self):
        self._items = []
        self.incr_version()

    def incr_version(self):
        global _version
        v = _version
        v[0] += 1
        self._version = v[0]
    if sys.implementation.name != 'pypy':

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._items)