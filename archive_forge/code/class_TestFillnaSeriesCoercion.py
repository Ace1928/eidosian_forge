from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
class TestFillnaSeriesCoercion(CoercionBase):
    method = 'fillna'

    @pytest.mark.xfail(reason='Test not implemented')
    def test_has_comprehensive_tests(self):
        raise NotImplementedError

    def _assert_fillna_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by fillna"""
        target = original.copy()
        res = target.fillna(value)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize('fill_val, fill_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, object)])
    def test_fillna_object(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass(['a', np.nan, 'c', 'd'], dtype=object)
        assert obj.dtype == object
        exp = klass(['a', fill_val, 'c', 'd'], dtype=object)
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_fillna_float64(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass([1.1, np.nan, 3.3, 4.4])
        assert obj.dtype == np.float64
        exp = klass([1.1, fill_val, 3.3, 4.4])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(1, np.complex128), (1.1, np.complex128), (1 + 1j, np.complex128), (True, object)])
    def test_fillna_complex128(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass([1 + 1j, np.nan, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128
        exp = klass([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'), (pd.Timestamp('2012-01-01', tz='US/Eastern'), object), (1, object), ('x', object)], ids=['datetime64', 'datetime64tz', 'object', 'object'])
    def test_fillna_datetime(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass([pd.Timestamp('2011-01-01'), pd.NaT, pd.Timestamp('2011-01-03'), pd.Timestamp('2011-01-04')])
        assert obj.dtype == 'datetime64[ns]'
        exp = klass([pd.Timestamp('2011-01-01'), fill_val, pd.Timestamp('2011-01-03'), pd.Timestamp('2011-01-04')])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]'), (pd.Timestamp('2012-01-01'), object), (pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 'datetime64[ns, US/Eastern]'), (1, object), ('x', object)])
    def test_fillna_datetime64tz(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        tz = 'US/Eastern'
        obj = klass([pd.Timestamp('2011-01-01', tz=tz), pd.NaT, pd.Timestamp('2011-01-03', tz=tz), pd.Timestamp('2011-01-04', tz=tz)])
        assert obj.dtype == 'datetime64[ns, US/Eastern]'
        if getattr(fill_val, 'tz', None) is None:
            fv = fill_val
        else:
            fv = fill_val.tz_convert(tz)
        exp = klass([pd.Timestamp('2011-01-01', tz=tz), fv, pd.Timestamp('2011-01-03', tz=tz), pd.Timestamp('2011-01-04', tz=tz)])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val', [1, 1.1, 1 + 1j, True, pd.Interval(1, 2, closed='left'), pd.Timestamp('2012-01-01', tz='US/Eastern'), pd.Timestamp('2012-01-01'), pd.Timedelta(days=1), pd.Period('2016-01-01', 'D')])
    def test_fillna_interval(self, index_or_series, fill_val):
        ii = pd.interval_range(1.0, 5.0, closed='right').insert(1, np.nan)
        assert isinstance(ii.dtype, pd.IntervalDtype)
        obj = index_or_series(ii)
        exp = index_or_series([ii[0], fill_val, ii[2], ii[3], ii[4]], dtype=object)
        fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_series_int64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_int64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_series_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_series_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.parametrize('fill_val', [1, 1.1, 1 + 1j, True, pd.Interval(1, 2, closed='left'), pd.Timestamp('2012-01-01', tz='US/Eastern'), pd.Timestamp('2012-01-01'), pd.Timedelta(days=1), pd.Period('2016-01-01', 'W')])
    def test_fillna_series_period(self, index_or_series, fill_val):
        pi = pd.period_range('2016-01-01', periods=4, freq='D').insert(1, pd.NaT)
        assert isinstance(pi.dtype, pd.PeriodDtype)
        obj = index_or_series(pi)
        exp = index_or_series([pi[0], fill_val, pi[2], pi[3], pi[4]], dtype=object)
        fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_period(self):
        raise NotImplementedError