import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class AbstractSet(Sized, Iterable[T_co], Container[T_co], extra=collections_abc.Set):
    __slots__ = ()