import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _get_stack_length(func):
    """Return function call stack length."""
    _func = func.__globals__.get(func.__name__, func)
    length = _count_wrappers(_func)
    return length