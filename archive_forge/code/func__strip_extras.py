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
def _strip_extras(t):
    """Strips Annotated, Required and NotRequired from a given type."""
    if isinstance(t, _AnnotatedAlias):
        return _strip_extras(t.__origin__)
    if hasattr(t, '__origin__') and t.__origin__ in (Required, NotRequired):
        return _strip_extras(t.__args__[0])
    if isinstance(t, typing._GenericAlias):
        stripped_args = tuple((_strip_extras(a) for a in t.__args__))
        if stripped_args == t.__args__:
            return t
        return t.copy_with(stripped_args)
    if hasattr(_types, 'GenericAlias') and isinstance(t, _types.GenericAlias):
        stripped_args = tuple((_strip_extras(a) for a in t.__args__))
        if stripped_args == t.__args__:
            return t
        return _types.GenericAlias(t.__origin__, stripped_args)
    if hasattr(_types, 'UnionType') and isinstance(t, _types.UnionType):
        stripped_args = tuple((_strip_extras(a) for a in t.__args__))
        if stripped_args == t.__args__:
            return t
        return functools.reduce(operator.or_, stripped_args)
    return t