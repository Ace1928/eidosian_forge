from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestNAObj:

    def _check_behavior(self, arr, expected):
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = libmissing.isnaobj(arr, inf_as_na=True)
        tm.assert_numpy_array_equal(result, expected)
        arr = np.atleast_2d(arr)
        expected = np.atleast_2d(expected)
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = libmissing.isnaobj(arr, inf_as_na=True)
        tm.assert_numpy_array_equal(result, expected)
        arr = arr.copy(order='F')
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = libmissing.isnaobj(arr, inf_as_na=True)
        tm.assert_numpy_array_equal(result, expected)

    def test_basic(self):
        arr = np.array([1, None, 'foo', -5.1, NaT, np.nan])
        expected = np.array([False, True, False, False, True, True])
        self._check_behavior(arr, expected)

    def test_non_obj_dtype(self):
        arr = np.array([1, 3, np.nan, 5], dtype=float)
        expected = np.array([False, False, True, False])
        self._check_behavior(arr, expected)

    def test_empty_arr(self):
        arr = np.array([])
        expected = np.array([], dtype=bool)
        self._check_behavior(arr, expected)

    def test_empty_str_inp(self):
        arr = np.array([''])
        expected = np.array([False])
        self._check_behavior(arr, expected)

    def test_empty_like(self):
        arr = np.empty_like([None])
        expected = np.array([True])
        self._check_behavior(arr, expected)