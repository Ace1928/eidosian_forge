import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def _has_varargs(func):
    if func not in signatures:
        return None
    sigs = signatures[func]
    checks = [check_varargs(sig) for sig in sigs]
    if all(checks):
        return True
    elif any(checks):
        return None
    return False