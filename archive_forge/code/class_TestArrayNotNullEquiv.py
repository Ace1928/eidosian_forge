from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
class TestArrayNotNullEquiv:

    @pytest.mark.parametrize('arr1, arr2', [(np.array([1, 2, 3]), np.array([1, 2, 3])), (np.array([1, 2, np.nan]), np.array([1, np.nan, 3])), (np.array([np.nan, 2, np.nan]), np.array([1, np.nan, np.nan]))])
    def test_equal(self, arr1, arr2):
        assert array_notnull_equiv(arr1, arr2)

    def test_some_not_equal(self):
        a = np.array([1, 2, 4])
        b = np.array([1, np.nan, 3])
        assert not array_notnull_equiv(a, b)

    def test_wrong_shape(self):
        a = np.array([[1, np.nan, np.nan, 4]])
        b = np.array([[1, 2], [np.nan, 4]])
        assert not array_notnull_equiv(a, b)

    @pytest.mark.parametrize('val1, val2, val3, null', [(np.datetime64('2000'), np.datetime64('2001'), np.datetime64('2002'), np.datetime64('NaT')), (1.0, 2.0, 3.0, np.nan), ('foo', 'bar', 'baz', None), ('foo', 'bar', 'baz', np.nan)])
    def test_types(self, val1, val2, val3, null):
        dtype = object if isinstance(val1, str) else None
        arr1 = np.array([val1, null, val3, null], dtype=dtype)
        arr2 = np.array([val1, val2, null, null], dtype=dtype)
        assert array_notnull_equiv(arr1, arr2)