import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar, Union, cast
from langchain_core._api.internal import is_caller_internal
class _beta_property(property):
    """A beta property."""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget, fset, fdel, doc)
        self.__orig_fget = fget
        self.__orig_fset = fset
        self.__orig_fdel = fdel

    def __get__(self, instance, owner=None):
        if instance is not None or owner is not None:
            emit_warning()
        return self.fget(instance)

    def __set__(self, instance, value):
        if instance is not None:
            emit_warning()
        return self.fset(instance, value)

    def __delete__(self, instance):
        if instance is not None:
            emit_warning()
        return self.fdel(instance)

    def __set_name__(self, owner, set_name):
        nonlocal _name
        if _name == '<lambda>':
            _name = set_name