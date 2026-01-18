from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestDatetimeArray(SharedTests):
    index_cls = DatetimeIndex
    array_cls = DatetimeArray
    scalar_type = Timestamp
    example_dtype = 'M8[ns]'

    @pytest.fixture
    def arr1d(self, tz_naive_fixture, freqstr):
        """
        Fixture returning DatetimeArray with parametrized frequency and
        timezones
        """
        tz = tz_naive_fixture
        dti = pd.date_range('2016-01-01 01:01:00', periods=5, freq=freqstr, tz=tz)
        dta = dti._data
        return dta

    def test_round(self, arr1d):
        dti = self.index_cls(arr1d)
        result = dti.round(freq='2min')
        expected = dti - pd.Timedelta(minutes=1)
        expected = expected._with_freq(None)
        tm.assert_index_equal(result, expected)
        dta = dti._data
        result = dta.round(freq='2min')
        expected = expected._data._with_freq(None)
        tm.assert_datetime_array_equal(result, expected)

    def test_array_interface(self, datetime_index):
        arr = datetime_index._data
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='datetime64[ns]')
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='datetime64[ns]', copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='datetime64[ns]')
        assert result is not expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='int64')
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)
        for dtype in ['float64', str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_array_object_dtype(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        expected = np.array(list(dti))
        result = np.array(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(dti, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_tz(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        expected = dti.asi8.view('M8[ns]')
        result = np.array(arr, dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='datetime64[ns]')
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='M8[ns]', copy=False)
        assert result.base is expected.base
        assert result.base is not None
        result = np.array(arr, dtype='datetime64[ns]', copy=False)
        assert result.base is expected.base
        assert result.base is not None

    def test_array_i8_dtype(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        expected = dti.asi8
        result = np.array(arr, dtype='i8')
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='i8', copy=False)
        assert result.base is not expected.base
        assert result.base is None

    def test_from_array_keeps_base(self):
        arr = np.array(['2000-01-01', '2000-01-02'], dtype='M8[ns]')
        dta = DatetimeArray._from_sequence(arr)
        assert dta._ndarray is arr
        dta = DatetimeArray._from_sequence(arr[:0])
        assert dta._ndarray.base is arr

    def test_from_dti(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        assert list(dti) == list(arr)
        dti2 = pd.Index(arr)
        assert isinstance(dti2, DatetimeIndex)
        assert list(dti2) == list(arr)

    def test_astype_object(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        asobj = arr.astype('O')
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == 'O'
        assert list(asobj) == list(dti)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_to_period(self, datetime_index, freqstr):
        dti = datetime_index
        arr = dti._data
        freqstr = freq_to_period_freqstr(1, freqstr)
        expected = dti.to_period(freq=freqstr)
        result = arr.to_period(freq=freqstr)
        assert isinstance(result, PeriodArray)
        tm.assert_equal(result, expected._data)

    def test_to_period_2d(self, arr1d):
        arr2d = arr1d.reshape(1, -1)
        warn = None if arr1d.tz is None else UserWarning
        with tm.assert_produces_warning(warn):
            result = arr2d.to_period('D')
            expected = arr1d.to_period('D').reshape(1, -1)
        tm.assert_period_array_equal(result, expected)

    @pytest.mark.parametrize('propname', DatetimeArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        dti = self.index_cls(arr1d)
        arr = arr1d
        assert dti.freq == arr.freq
        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('propname', DatetimeArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        dti = self.index_cls(arr1d)
        arr = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, arr1d, fixed_now_ts):
        arr = arr1d
        dti = self.index_cls(arr1d)
        now = fixed_now_ts.tz_localize(dti.tz)
        result = arr.take([-1, 1], allow_fill=True, fill_value=now)
        assert result[0] == now
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=now - now)
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=Period('2014Q1'))
        tz = None if dti.tz is not None else 'US/Eastern'
        now = fixed_now_ts.tz_localize(tz)
        msg = 'Cannot compare tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=now)
        value = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)
        value = np.timedelta64('NaT', 'ns')
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)
        if arr.tz is not None:
            value = fixed_now_ts.tz_localize('Australia/Melbourne')
            result = arr.take([-1, 1], allow_fill=True, fill_value=value)
            expected = arr.take([-1, 1], allow_fill=True, fill_value=value.tz_convert(arr.dtype.tz))
            tm.assert_equal(result, expected)

    def test_concat_same_type_invalid(self, arr1d):
        arr = arr1d
        if arr.tz is None:
            other = arr.tz_localize('UTC')
        else:
            other = arr.tz_localize(None)
        with pytest.raises(ValueError, match='to_concat must have the same'):
            arr._concat_same_type([arr, other])

    def test_concat_same_type_different_freq(self, unit):
        a = pd.date_range('2000', periods=2, freq='D', tz='US/Central', unit=unit)._data
        b = pd.date_range('2000', periods=2, freq='h', tz='US/Central', unit=unit)._data
        result = DatetimeArray._concat_same_type([a, b])
        expected = pd.to_datetime(['2000-01-01 00:00:00', '2000-01-02 00:00:00', '2000-01-01 00:00:00', '2000-01-01 01:00:00']).tz_localize('US/Central').as_unit(unit)._data
        tm.assert_datetime_array_equal(result, expected)

    def test_strftime(self, arr1d):
        arr = arr1d
        result = arr.strftime('%Y %b')
        expected = np.array([ts.strftime('%Y %b') for ts in arr], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime_nat(self):
        arr = DatetimeIndex(['2019-01-01', NaT])._data
        result = arr.strftime('%Y-%m-%d')
        expected = np.array(['2019-01-01', np.nan], dtype=object)
        tm.assert_numpy_array_equal(result, expected)