from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class _VArgsWrapper:
    """
    A wrapper around a Callable. It delegates `__call__` to the Callable.
    If the Callable has a `__get__`, that is also delegate and the resulting function is wrapped.
    Otherwise, we use the original function mirroring the behaviour without a __get__.
    We also have the visit_wrapper attribute to be used by Transformers.
    """
    base_func: Callable

    def __init__(self, func: Callable, visit_wrapper: Callable[[Callable, str, list, Any], Any]):
        if isinstance(func, _VArgsWrapper):
            func = func.base_func
        self.base_func = func
        self.visit_wrapper = visit_wrapper
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.base_func(*args, **kwargs)

    def __get__(self, instance, owner=None):
        try:
            g = type(self.base_func).__get__
        except AttributeError:
            return self
        else:
            return _VArgsWrapper(g(self.base_func, instance, owner), self.visit_wrapper)

    def __set_name__(self, owner, name):
        try:
            f = type(self.base_func).__set_name__
        except AttributeError:
            return
        else:
            f(self.base_func, owner, name)