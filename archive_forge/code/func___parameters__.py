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
@property
def __parameters__(self):
    return tuple((tp for tp in self.__args__ if isinstance(tp, (typing.TypeVar, ParamSpec))))