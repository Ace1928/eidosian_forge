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
class _Immutable:
    """Mixin to indicate that object should not be copied."""
    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self