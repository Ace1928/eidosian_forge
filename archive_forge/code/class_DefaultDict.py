import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class DefaultDict(collections.defaultdict, MutableMapping[KT, VT], extra=collections.defaultdict):
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls._gorg is DefaultDict:
            return collections.defaultdict(*args, **kwds)
        return _generic_new(collections.defaultdict, cls, *args, **kwds)