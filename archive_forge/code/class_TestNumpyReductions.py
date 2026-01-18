from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
class TestNumpyReductions:

    def test_multiply(self, values_for_np_reduce, box_with_array, request):
        box = box_with_array
        values = values_for_np_reduce
        with tm.assert_produces_warning(None):
            obj = box(values)
        if isinstance(values, pd.core.arrays.SparseArray):
            mark = pytest.mark.xfail(reason="SparseArray has no 'prod'")
            request.applymarker(mark)
        if values.dtype.kind in 'iuf':
            result = np.multiply.reduce(obj)
            if box is pd.DataFrame:
                expected = obj.prod(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                expected = obj._values.prod()
                assert result == expected
            else:
                expected = obj.prod()
                assert result == expected
        else:
            msg = '|'.join(['does not support reduction', 'unsupported operand type', "ufunc 'multiply' cannot use operands"])
            with pytest.raises(TypeError, match=msg):
                np.multiply.reduce(obj)

    def test_add(self, values_for_np_reduce, box_with_array):
        box = box_with_array
        values = values_for_np_reduce
        with tm.assert_produces_warning(None):
            obj = box(values)
        if values.dtype.kind in 'miuf':
            result = np.add.reduce(obj)
            if box is pd.DataFrame:
                expected = obj.sum(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                expected = obj._values.sum()
                assert result == expected
            else:
                expected = obj.sum()
                assert result == expected
        else:
            msg = '|'.join(['does not support reduction', 'unsupported operand type', "ufunc 'add' cannot use operands"])
            with pytest.raises(TypeError, match=msg):
                np.add.reduce(obj)

    def test_max(self, values_for_np_reduce, box_with_array):
        box = box_with_array
        values = values_for_np_reduce
        same_type = True
        if box is pd.Index and values.dtype.kind in ['i', 'f']:
            same_type = False
        with tm.assert_produces_warning(None):
            obj = box(values)
        result = np.maximum.reduce(obj)
        if box is pd.DataFrame:
            expected = obj.max(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected = values[1]
            assert result == expected
            if same_type:
                assert type(result) == type(expected)

    def test_min(self, values_for_np_reduce, box_with_array):
        box = box_with_array
        values = values_for_np_reduce
        same_type = True
        if box is pd.Index and values.dtype.kind in ['i', 'f']:
            same_type = False
        with tm.assert_produces_warning(None):
            obj = box(values)
        result = np.minimum.reduce(obj)
        if box is pd.DataFrame:
            expected = obj.min(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected = values[0]
            assert result == expected
            if same_type:
                assert type(result) == type(expected)