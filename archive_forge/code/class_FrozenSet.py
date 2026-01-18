import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class FrozenSet(frozenset, AbstractSet[T_co], extra=frozenset):
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls._gorg is FrozenSet:
            raise TypeError('Type FrozenSet cannot be instantiated; use frozenset() instead')
        return _generic_new(frozenset, cls, *args, **kwds)