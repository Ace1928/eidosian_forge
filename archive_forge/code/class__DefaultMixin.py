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
class _DefaultMixin:
    """Mixin for TypeVarLike defaults."""
    __slots__ = ()
    __init__ = _set_default