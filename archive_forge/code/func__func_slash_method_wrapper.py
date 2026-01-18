from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def _func_slash_method_wrapper(f, name=None):
    if name is None:
        name = f.__name__

    def func(self, *args, **kwargs):
        try:
            return getattr(self, name)(*args, **kwargs)
        except AttributeError:
            return f(self, *args, **kwargs)
    func.__name__ = name
    func.__doc__ = f.__doc__
    return func