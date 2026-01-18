import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
import pandas._testing as tm
class TestAstypeOverflowSafe:

    def test_pass_non_dt64_array(self):
        arr = np.arange(5)
        dtype = np.dtype('M8[ns]')
        msg = 'astype_overflowsafe values.dtype and dtype must be either both-datetime64 or both-timedelta64'
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_pass_non_dt64_dtype(self):
        arr = np.arange(5, dtype='i8').view('M8[D]')
        dtype = np.dtype('m8[ns]')
        msg = 'astype_overflowsafe values.dtype and dtype must be either both-datetime64 or both-timedelta64'
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_astype_overflowsafe_dt64(self):
        dtype = np.dtype('M8[ns]')
        dt = np.datetime64('2262-04-05', 'D')
        arr = dt + np.arange(10, dtype='m8[D]')
        wrong = arr.astype(dtype)
        roundtrip = wrong.astype(arr.dtype)
        assert not (wrong == roundtrip).all()
        msg = 'Out of bounds nanosecond timestamp'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            astype_overflowsafe(arr, dtype)
        dtype2 = np.dtype('M8[us]')
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_overflowsafe_td64(self):
        dtype = np.dtype('m8[ns]')
        dt = np.datetime64('2262-04-05', 'D')
        arr = dt + np.arange(10, dtype='m8[D]')
        arr = arr.view('m8[D]')
        wrong = arr.astype(dtype)
        roundtrip = wrong.astype(arr.dtype)
        assert not (wrong == roundtrip).all()
        msg = 'Cannot convert 106752 days to timedelta64\\[ns\\] without overflow'
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            astype_overflowsafe(arr, dtype)
        dtype2 = np.dtype('m8[us]')
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_overflowsafe_disallow_rounding(self):
        arr = np.array([-1500, 1500], dtype='M8[ns]')
        dtype = np.dtype('M8[us]')
        msg = "Cannot losslessly cast '-1500 ns' to us"
        with pytest.raises(ValueError, match=msg):
            astype_overflowsafe(arr, dtype, round_ok=False)
        result = astype_overflowsafe(arr, dtype, round_ok=True)
        expected = arr.astype(dtype)
        tm.assert_numpy_array_equal(result, expected)