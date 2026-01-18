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
@_ExtensionsSpecialForm
def Concatenate(self, parameters):
    """Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """
    return _concatenate_getitem(self, parameters)