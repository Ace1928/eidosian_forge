from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.filterwarnings("ignore:Series.fillna with 'method' is deprecated:FutureWarning")
class TestFillnaPad:

    def test_fillna_bug(self):
        ser = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'])
        filled = ser.fillna(method='ffill')
        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ser.index)
        tm.assert_series_equal(filled, expected)
        filled = ser.fillna(method='bfill')
        expected = Series([1.0, 1.0, 3.0, 3.0, np.nan], ser.index)
        tm.assert_series_equal(filled, expected)

    def test_ffill(self):
        ts = Series([0.0, 1.0, 2.0, 3.0, 4.0], index=date_range('2020-01-01', periods=5))
        ts.iloc[2] = np.nan
        tm.assert_series_equal(ts.ffill(), ts.fillna(method='ffill'))

    def test_ffill_mixed_dtypes_without_missing_data(self):
        series = Series([datetime(2015, 1, 1, tzinfo=pytz.utc), 1])
        result = series.ffill()
        tm.assert_series_equal(series, result)

    def test_bfill(self):
        ts = Series([0.0, 1.0, 2.0, 3.0, 4.0], index=date_range('2020-01-01', periods=5))
        ts.iloc[2] = np.nan
        tm.assert_series_equal(ts.bfill(), ts.fillna(method='bfill'))

    def test_pad_nan(self):
        x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'], dtype=float)
        return_value = x.fillna(method='pad', inplace=True)
        assert return_value is None
        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ['z', 'a', 'b', 'c', 'd'], dtype=float)
        tm.assert_series_equal(x[1:], expected[1:])
        assert np.isnan(x.iloc[0]), np.isnan(expected.iloc[0])

    def test_series_fillna_limit(self):
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)
        result = s[:2].reindex(index)
        result = result.fillna(method='pad', limit=5)
        expected = s[:2].reindex(index).fillna(method='pad')
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)
        result = s[-2:].reindex(index)
        result = result.fillna(method='bfill', limit=5)
        expected = s[-2:].reindex(index).fillna(method='backfill')
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_series_pad_backfill_limit(self):
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)
        result = s[:2].reindex(index, method='pad', limit=5)
        expected = s[:2].reindex(index).fillna(method='pad')
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)
        result = s[-2:].reindex(index, method='backfill', limit=5)
        expected = s[-2:].reindex(index).fillna(method='backfill')
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_fillna_int(self):
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        return_value = ser.fillna(method='ffill', inplace=True)
        assert return_value is None
        tm.assert_series_equal(ser.fillna(method='ffill', inplace=False), ser)

    def test_datetime64tz_fillna_round_issue(self):
        data = Series([NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc)])
        filled = data.bfill()
        expected = Series([datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc), datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc), datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc)])
        tm.assert_series_equal(filled, expected)

    def test_fillna_parr(self):
        dti = date_range(Timestamp.max - Timedelta(nanoseconds=10), periods=5, freq='ns')
        ser = Series(dti.to_period('ns'))
        ser[2] = NaT
        arr = period_array([Timestamp('2262-04-11 23:47:16.854775797'), Timestamp('2262-04-11 23:47:16.854775798'), Timestamp('2262-04-11 23:47:16.854775798'), Timestamp('2262-04-11 23:47:16.854775800'), Timestamp('2262-04-11 23:47:16.854775801')], freq='ns')
        expected = Series(arr)
        filled = ser.ffill()
        tm.assert_series_equal(filled, expected)

    @pytest.mark.parametrize('func', ['pad', 'backfill'])
    def test_pad_backfill_deprecated(self, func):
        ser = Series([1, 2, 3])
        with tm.assert_produces_warning(FutureWarning):
            getattr(ser, func)()