from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
class function:
    """wrapper class for numpy functions

    Same as method, but the name is used for referencing numpy functions
    """

    def __init__(self, name_or_function, *args, function_label=None, **kwargs):
        if callable(name_or_function):
            self.name = function_label if function_label is not None else name_or_function.__name__
            self.func = name_or_function
        else:
            self.name = name_or_function if function_label is None else function_label
            self.func = getattr(np, name_or_function)
            if self.func is None:
                raise AttributeError(f"module 'numpy' has no attribute named '{self.name}'")
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        all_args = merge_args(self.args, args)
        all_kwargs = {**self.kwargs, **kwargs}
        return self.func(*all_args, **all_kwargs)

    def __repr__(self):
        return f'function_{self.name}'