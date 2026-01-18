from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _ensure_3args(func):
    if func is None:
        return None
    self_arg = 1 if inspect.ismethod(func) else 0
    if hasattr(inspect, 'getfullargspec'):
        args = inspect.getfullargspec(func)[0]
    else:
        args = inspect.getargspec(func)[0]
    if len(args) == 3 + self_arg:
        return func
    if len(args) == 2 + self_arg:
        return lambda x, params=(), backend=math: func(x, params)
    elif len(args) == 1 + self_arg:
        return lambda x, params=(), backend=math: func(x)
    else:
        raise ValueError('Incorrect numer of arguments')