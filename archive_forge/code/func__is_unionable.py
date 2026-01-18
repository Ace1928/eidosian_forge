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
def _is_unionable(obj):
    """Corresponds to is_unionable() in unionobject.c in CPython."""
    return obj is None or isinstance(obj, (type, _types.GenericAlias, _types.UnionType, TypeAliasType))