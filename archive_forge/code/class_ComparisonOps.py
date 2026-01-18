import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil
class ComparisonOps(BaseOpsUtil):

    def _compare_other(self, data, op, other):
        result = pd.Series(op(data, other))
        expected = pd.Series(op(data._data, other), dtype='boolean')
        expected[data._mask] = pd.NA
        tm.assert_series_equal(result, expected)
        ser = pd.Series(data)
        result = op(ser, other)
        expected = op(pd.Series(data._data), other).astype('boolean')
        expected[data._mask] = pd.NA
        tm.assert_series_equal(result, expected)

    def test_scalar(self, other, comparison_op, dtype):
        op = comparison_op
        left = pd.array([1, 0, None], dtype=dtype)
        result = op(left, other)
        if other is pd.NA:
            expected = pd.array([None, None, None], dtype='boolean')
        else:
            values = op(left._data, other)
            expected = pd.arrays.BooleanArray(values, left._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)
        result[0] = pd.NA
        tm.assert_extension_array_equal(left, pd.array([1, 0, None], dtype=dtype))