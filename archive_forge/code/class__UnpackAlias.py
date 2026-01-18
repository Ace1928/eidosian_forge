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
class _UnpackAlias(typing._GenericAlias, _root=True):
    __class__ = typing.TypeVar