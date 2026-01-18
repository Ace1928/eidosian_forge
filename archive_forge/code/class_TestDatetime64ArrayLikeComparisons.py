from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
class TestDatetime64ArrayLikeComparisons:

    def test_compare_zerodim(self, tz_naive_fixture, box_with_array):
        tz = tz_naive_fixture
        box = box_with_array
        dti = date_range('20130101', periods=3, tz=tz)
        other = np.array(dti.to_numpy()[0])
        dtarr = tm.box_expected(dti, box)
        xbox = get_upcast_box(dtarr, other, True)
        result = dtarr <= other
        expected = np.array([True, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('other', ['foo', -1, 99, 4.0, object(), timedelta(days=2), datetime(2001, 1, 1).date(), None, np.nan])
    def test_dt64arr_cmp_scalar_invalid(self, other, tz_naive_fixture, box_with_array):
        tz = tz_naive_fixture
        rng = date_range('1/1/2000', periods=10, tz=tz)
        dtarr = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize('other', [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.timedelta_range('1ns', periods=10).array, np.array(pd.timedelta_range('1ns', periods=10)), list(pd.timedelta_range('1ns', periods=10)), pd.timedelta_range('1 Day', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
    def test_dt64arr_cmp_arraylike_invalid(self, other, tz_naive_fixture, box_with_array):
        tz = tz_naive_fixture
        dta = date_range('1970-01-01', freq='ns', periods=10, tz=tz)._data
        obj = tm.box_expected(dta, box_with_array)
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture):
        tz = tz_naive_fixture
        dta = date_range('1970-01-01', freq='h', periods=5, tz=tz)._data
        other = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result = dta == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)
        msg = 'Invalid comparison between|Cannot compare type|not supported between'
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture, box_with_array):
        tz = tz_naive_fixture
        box = box_with_array
        ts = Timestamp('2021-01-01', tz=tz)
        ser = Series([ts, NaT])
        obj = tm.box_expected(ser, box)
        xbox = get_upcast_box(obj, ts, True)
        expected = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)
        result = obj == ts
        tm.assert_equal(result, expected)