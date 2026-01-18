import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def _is_valid_args(func, args, kwargs):
    """ Like ``is_valid_args`` for builtins in our ``signatures`` registry"""
    if func not in signatures:
        return None
    sigs = signatures[func]
    return any((check_valid(sig, args, kwargs) for sig in sigs))