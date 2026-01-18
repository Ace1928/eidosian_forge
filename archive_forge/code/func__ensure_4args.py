from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
def _ensure_4args(func):
    """ Conditionally wrap function to ensure 4 input arguments

    Parameters
    ----------
    func: callable
        with two, three or four positional arguments

    Returns
    -------
    callable which possibly ignores 0, 1 or 2 positional arguments

    """
    if func is None:
        return None
    if isinstance(func, _Blessed):
        return func
    self_arg = 1 if inspect.ismethod(func) else 0
    if hasattr(inspect, 'getfullargspec'):
        args = inspect.getfullargspec(func)[0]
    else:
        args = inspect.getargspec(func)[0]
    if len(args) == 4 + self_arg:
        return func
    if len(args) == 3 + self_arg:
        return lambda x, y, p=(), backend=math: func(x, y, p)
    elif len(args) == 2 + self_arg:
        return lambda x, y, p=(), backend=math: func(x, y)
    else:
        raise ValueError('Incorrect numer of arguments')