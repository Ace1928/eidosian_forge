import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _dispatch(func=None, **kwargs):
    if func is None:
        return partial(_dispatch, **kwargs)
    dispatched_func = _orig_dispatch(func, **kwargs)
    func.__doc__ = dispatched_func.__doc__
    return func