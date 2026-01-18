import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@runtime_checkable
class SupportsInt(Protocol):
    """An ABC with one abstract method __int__."""
    __slots__ = ()

    @abc.abstractmethod
    def __int__(self) -> int:
        pass