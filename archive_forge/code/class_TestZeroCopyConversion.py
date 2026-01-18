import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
class TestZeroCopyConversion:
    """
    Tests that zero-copy conversion works with some types.
    """

    def test_zero_copy_success(self):
        result = pa.array([0, 1, 2]).to_pandas(zero_copy_only=True)
        npt.assert_array_equal(result, [0, 1, 2])

    def test_zero_copy_dictionaries(self):
        arr = pa.DictionaryArray.from_arrays(np.array([0, 0]), np.array([5], dtype='int64'))
        result = arr.to_pandas(zero_copy_only=True)
        values = pd.Categorical([5, 5])
        tm.assert_series_equal(pd.Series(result), pd.Series(values), check_names=False)

    def test_zero_copy_timestamp(self):
        arr = np.array(['2007-07-13'], dtype='datetime64[ns]')
        result = pa.array(arr).to_pandas(zero_copy_only=True)
        npt.assert_array_equal(result, arr)

    def test_zero_copy_duration(self):
        arr = np.array([1], dtype='timedelta64[ns]')
        result = pa.array(arr).to_pandas(zero_copy_only=True)
        npt.assert_array_equal(result, arr)

    def check_zero_copy_failure(self, arr):
        with pytest.raises(pa.ArrowInvalid):
            arr.to_pandas(zero_copy_only=True)

    def test_zero_copy_failure_on_object_types(self):
        self.check_zero_copy_failure(pa.array(['A', 'B', 'C']))

    def test_zero_copy_failure_with_int_when_nulls(self):
        self.check_zero_copy_failure(pa.array([0, 1, None]))

    def test_zero_copy_failure_with_float_when_nulls(self):
        self.check_zero_copy_failure(pa.array([0.0, 1.0, None]))

    def test_zero_copy_failure_on_bool_types(self):
        self.check_zero_copy_failure(pa.array([True, False]))

    def test_zero_copy_failure_on_list_types(self):
        arr = pa.array([[1, 2], [8, 9]], type=pa.list_(pa.int64()))
        self.check_zero_copy_failure(arr)

    def test_zero_copy_failure_on_timestamp_with_nulls(self):
        arr = np.array([1, None], dtype='datetime64[ns]')
        self.check_zero_copy_failure(pa.array(arr))

    def test_zero_copy_failure_on_duration_with_nulls(self):
        arr = np.array([1, None], dtype='timedelta64[ns]')
        self.check_zero_copy_failure(pa.array(arr))