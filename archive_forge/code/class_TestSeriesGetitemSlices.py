from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
class TestSeriesGetitemSlices:

    def test_getitem_partial_str_slice_with_datetimeindex(self):
        arr = date_range('1/1/2008', '1/1/2009')
        ser = arr.to_series()
        result = ser['2008']
        rng = date_range(start='2008-01-01', end='2008-12-31')
        expected = Series(rng, index=rng)
        tm.assert_series_equal(result, expected)

    def test_getitem_slice_strings_with_datetimeindex(self):
        idx = DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/3/2000', '1/4/2000'])
        ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)
        result = ts['1/2/2000':]
        expected = ts[1:]
        tm.assert_series_equal(result, expected)
        result = ts['1/2/2000':'1/3/2000']
        expected = ts[1:4]
        tm.assert_series_equal(result, expected)

    def test_getitem_partial_str_slice_with_timedeltaindex(self):
        rng = timedelta_range('1 day 10:11:12', freq='h', periods=500)
        ser = Series(np.arange(len(rng)), index=rng)
        result = ser['5 day':'6 day']
        expected = ser.iloc[86:134]
        tm.assert_series_equal(result, expected)
        result = ser['5 day':]
        expected = ser.iloc[86:]
        tm.assert_series_equal(result, expected)
        result = ser[:'6 day']
        expected = ser.iloc[:134]
        tm.assert_series_equal(result, expected)

    def test_getitem_partial_str_slice_high_reso_with_timedeltaindex(self):
        rng = timedelta_range('1 day 10:11:12', freq='us', periods=2000)
        ser = Series(np.arange(len(rng)), index=rng)
        result = ser['1 day 10:11:12':]
        expected = ser.iloc[0:]
        tm.assert_series_equal(result, expected)
        result = ser['1 day 10:11:12.001':]
        expected = ser.iloc[1000:]
        tm.assert_series_equal(result, expected)
        result = ser['1 days, 10:11:12.001001']
        assert result == ser.iloc[1001]

    def test_getitem_slice_2d(self, datetime_series):
        with pytest.raises(ValueError, match='Multi-dimensional indexing'):
            datetime_series[:, np.newaxis]

    def test_getitem_median_slice_bug(self):
        index = date_range('20090415', '20090519', freq='2B')
        ser = Series(np.random.default_rng(2).standard_normal(13), index=index)
        indexer = [slice(6, 7, None)]
        msg = 'Indexing with a single-item list'
        with pytest.raises(ValueError, match=msg):
            ser[indexer]
        result = ser[indexer[0],]
        expected = ser[indexer[0]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('slc, positions', [[slice(date(2018, 1, 1), None), [0, 1, 2]], [slice(date(2019, 1, 2), None), [2]], [slice(date(2020, 1, 1), None), []], [slice(None, date(2020, 1, 1)), [0, 1, 2]], [slice(None, date(2019, 1, 1)), [0]]])
    def test_getitem_slice_date(self, slc, positions):
        ser = Series([0, 1, 2], DatetimeIndex(['2019-01-01', '2019-01-01T06:00:00', '2019-01-02']))
        result = ser[slc]
        expected = ser.take(positions)
        tm.assert_series_equal(result, expected)

    def test_getitem_slice_float_raises(self, datetime_series):
        msg = 'cannot do slice indexing on DatetimeIndex with these indexers \\[{key}\\] of type float'
        with pytest.raises(TypeError, match=msg.format(key='4\\.0')):
            datetime_series[4.0:10.0]
        with pytest.raises(TypeError, match=msg.format(key='4\\.5')):
            datetime_series[4.5:10.0]

    def test_getitem_slice_bug(self):
        ser = Series(range(10), index=list(range(10)))
        result = ser[-12:]
        tm.assert_series_equal(result, ser)
        result = ser[-7:]
        tm.assert_series_equal(result, ser[3:])
        result = ser[:-12]
        tm.assert_series_equal(result, ser[:0])

    def test_getitem_slice_integers(self):
        ser = Series(np.random.default_rng(2).standard_normal(8), index=[2, 4, 6, 8, 10, 12, 14, 16])
        result = ser[:4]
        expected = Series(ser.values[:4], index=[2, 4, 6, 8])
        tm.assert_series_equal(result, expected)