import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
def _generic_new(base_cls, cls, *args, **kwds):
    if cls.__origin__ is None:
        if base_cls.__new__ is object.__new__ and cls.__init__ is not object.__init__:
            return base_cls.__new__(cls)
        else:
            return base_cls.__new__(cls, *args, **kwds)
    else:
        origin = cls._gorg
        if base_cls.__new__ is object.__new__ and cls.__init__ is not object.__init__:
            obj = base_cls.__new__(origin)
        else:
            obj = base_cls.__new__(origin, *args, **kwds)
        try:
            obj.__orig_class__ = cls
        except AttributeError:
            pass
        obj.__init__(*args, **kwds)
        return obj