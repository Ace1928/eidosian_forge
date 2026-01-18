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
class TestOps:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.x = array([[[nan, nan, 2.0, nan], [nan, 5.0, 6.0, nan], [8.0, 9.0, 10.0, nan]], [[nan, 13.0, 14.0, 15.0], [nan, 17.0, 18.0, nan], [nan, 21.0, nan, nan]]])

    def test_first(self):
        expected_results = [array([[nan, 13, 2, 15], [nan, 5, 6, nan], [8, 9, 10, nan]]), array([[8, 5, 2, nan], [nan, 13, 14, 15]]), array([[2, 5, 8], [13, 17, 21]])]
        for axis, expected in zip([0, 1, 2, -3, -2, -1], 2 * expected_results):
            actual = first(self.x, axis)
            assert_array_equal(expected, actual)
        expected = self.x[0]
        actual = first(self.x, axis=0, skipna=False)
        assert_array_equal(expected, actual)
        expected = self.x[..., 0]
        actual = first(self.x, axis=-1, skipna=False)
        assert_array_equal(expected, actual)
        with pytest.raises(IndexError, match='out of bounds'):
            first(self.x, 3)

    def test_last(self):
        expected_results = [array([[nan, 13, 14, 15], [nan, 17, 18, nan], [8, 21, 10, nan]]), array([[8, 9, 10, nan], [nan, 21, 18, 15]]), array([[2, 6, 10], [15, 18, 21]])]
        for axis, expected in zip([0, 1, 2, -3, -2, -1], 2 * expected_results):
            actual = last(self.x, axis)
            assert_array_equal(expected, actual)
        expected = self.x[-1]
        actual = last(self.x, axis=0, skipna=False)
        assert_array_equal(expected, actual)
        expected = self.x[..., -1]
        actual = last(self.x, axis=-1, skipna=False)
        assert_array_equal(expected, actual)
        with pytest.raises(IndexError, match='out of bounds'):
            last(self.x, 3)

    def test_count(self):
        assert 12 == count(self.x)
        expected = array([[1, 2, 3], [3, 2, 1]])
        assert_array_equal(expected, count(self.x, axis=-1))
        assert 1 == count(np.datetime64('2000-01-01'))

    def test_where_type_promotion(self):
        result = where([True, False], [1, 2], ['a', 'b'])
        assert_array_equal(result, np.array([1, 'b'], dtype=object))
        result = where([True, False], np.array([1, 2], np.float32), np.nan)
        assert result.dtype == np.float32
        assert_array_equal(result, np.array([1, np.nan], dtype=np.float32))

    def test_stack_type_promotion(self):
        result = stack([1, 'b'])
        assert_array_equal(result, np.array([1, 'b'], dtype=object))

    def test_concatenate_type_promotion(self):
        result = concatenate([[1], ['b']])
        assert_array_equal(result, np.array([1, 'b'], dtype=object))

    @pytest.mark.filterwarnings('error')
    def test_all_nan_arrays(self):
        assert np.isnan(mean([np.nan, np.nan]))