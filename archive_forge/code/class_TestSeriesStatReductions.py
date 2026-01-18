import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSeriesStatReductions:

    def _check_stat_op(self, name, alternate, string_series_, check_objects=False, check_allna=False):
        with pd.option_context('use_bottleneck', False):
            f = getattr(Series, name)
            string_series_[5:15] = np.nan
            if name not in ['max', 'min', 'mean', 'median', 'std']:
                ds = Series(date_range('1/1/2001', periods=10))
                msg = f"does not support reduction '{name}'"
                with pytest.raises(TypeError, match=msg):
                    f(ds)
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))
            allna = string_series_ * np.nan
            if check_allna:
                assert np.isnan(f(allna))
            s = Series([1, 2, 3, None, 5])
            f(s)
            items = [0]
            items.extend(range(2 ** 40, 2 ** 40 + 1000))
            s = Series(items, dtype='int64')
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))
            if check_objects:
                s = Series(pd.bdate_range('1/1/2000', periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp
            if name not in ['sum', 'min', 'max']:
                with pytest.raises(TypeError, match=None):
                    f(Series(list('abc')))
            msg = 'No axis named 1 for object type Series'
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)
            if 'numeric_only' in inspect.getfullargspec(f).args:
                f(string_series_, numeric_only=True)

    def test_sum(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('sum', np.sum, string_series, check_allna=False)

    def test_mean(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('mean', np.mean, string_series)

    def test_median(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('median', np.median, string_series)
        int_ts = Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('prod', np.prod, string_series)

    def test_min(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('min', np.min, string_series, check_objects=True)

    def test_max(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('max', np.max, string_series, check_objects=True)

    def test_var_std(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        datetime_series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        alt = lambda x: np.std(x, ddof=1)
        self._check_stat_op('std', alt, string_series)
        alt = lambda x: np.var(x, ddof=1)
        self._check_stat_op('var', alt, string_series)
        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)
        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)
        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        datetime_series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op('sem', alt, string_series)
        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(len(datetime_series.values))
        tm.assert_almost_equal(result, expected)
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    def test_skew(self):
        sp_stats = pytest.importorskip('scipy.stats')
        string_series = Series(range(20), dtype=np.float64, name='series')
        alt = lambda x: sp_stats.skew(x, bias=False)
        self._check_stat_op('skew', alt, string_series)
        min_N = 3
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                assert 0 == s.skew()
                assert isinstance(s.skew(), np.float64)
                assert (df.skew() == 0).all()

    def test_kurt(self):
        sp_stats = pytest.importorskip('scipy.stats')
        string_series = Series(range(20), dtype=np.float64, name='series')
        alt = lambda x: sp_stats.kurtosis(x, bias=False)
        self._check_stat_op('kurt', alt, string_series)

    def test_kurt_corner(self):
        min_N = 4
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                assert 0 == s.kurt()
                assert isinstance(s.kurt(), np.float64)
                assert (df.kurt() == 0).all()