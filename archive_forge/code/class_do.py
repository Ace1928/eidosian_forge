from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
class do:

    def __init__(self, meth, *args, **kwargs):
        self.meth = meth
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        kwargs = self.kwargs.copy()
        if 'func' in self.kwargs:
            kwargs['func'] = getattr(np, kwargs['func'])
        return getattr(obj, self.meth)(*self.args, **kwargs)

    def __repr__(self):
        return f'obj.{self.meth}(*{self.args}, **{self.kwargs})'