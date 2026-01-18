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
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
class TestPeriodArray(SharedTests):
    index_cls = PeriodIndex
    array_cls = PeriodArray
    scalar_type = Period
    example_dtype = PeriodIndex([], freq='W').dtype

    @pytest.fixture
    def arr1d(self, period_index):
        """
        Fixture returning DatetimeArray from parametrized PeriodIndex objects
        """
        return period_index._data

    def test_from_pi(self, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        assert list(arr) == list(pi)
        pi2 = pd.Index(arr)
        assert isinstance(pi2, PeriodIndex)
        assert list(pi2) == list(arr)

    def test_astype_object(self, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        asobj = arr.astype('O')
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == 'O'
        assert list(asobj) == list(pi)

    def test_take_fill_valid(self, arr1d):
        arr = arr1d
        value = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)
        value = np.timedelta64('NaT', 'ns')
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)

    @pytest.mark.parametrize('how', ['S', 'E'])
    def test_to_timestamp(self, how, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        expected = DatetimeIndex(pi.to_timestamp(how=how))._data
        result = arr.to_timestamp(how=how)
        assert isinstance(result, DatetimeArray)
        tm.assert_equal(result, expected)

    def test_to_timestamp_roundtrip_bday(self):
        dta = pd.date_range('2021-10-18', periods=3, freq='B')._data
        parr = dta.to_period()
        result = parr.to_timestamp()
        assert result.freq == 'B'
        tm.assert_extension_array_equal(result, dta)
        dta2 = dta[::2]
        parr2 = dta2.to_period()
        result2 = parr2.to_timestamp()
        assert result2.freq == '2B'
        tm.assert_extension_array_equal(result2, dta2)
        parr3 = dta.to_period('2B')
        result3 = parr3.to_timestamp()
        assert result3.freq == 'B'
        tm.assert_extension_array_equal(result3, dta)

    def test_to_timestamp_out_of_bounds(self):
        pi = pd.period_range('1500', freq='Y', periods=3)
        msg = 'Out of bounds nanosecond timestamp: 1500-01-01 00:00:00'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi.to_timestamp()
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi._data.to_timestamp()

    @pytest.mark.parametrize('propname', PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        pi = self.index_cls(arr1d)
        arr = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('propname', PeriodArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        pi = self.index_cls(arr1d)
        arr = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, arr1d):
        arr = arr1d
        result = np.asarray(arr)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='int64')
        tm.assert_numpy_array_equal(result, arr.asi8)
        msg = "float\\(\\) argument must be a string or a( real)? number, not 'Period'"
        with pytest.raises(TypeError, match=msg):
            np.asarray(arr, dtype='float64')
        result = np.asarray(arr, dtype='S20')
        expected = np.asarray(arr).astype('S20')
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime(self, arr1d):
        arr = arr1d
        result = arr.strftime('%Y')
        expected = np.array([per.strftime('%Y') for per in arr], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime_nat(self):
        arr = PeriodArray(PeriodIndex(['2019-01-01', NaT], dtype='period[D]'))
        result = arr.strftime('%Y-%m-%d')
        expected = np.array(['2019-01-01', np.nan], dtype=object)
        tm.assert_numpy_array_equal(result, expected)