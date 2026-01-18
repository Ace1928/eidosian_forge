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
class TestSeriesGetitemScalars:

    def test_getitem_object_index_float_string(self):
        ser = Series([1] * 4, index=Index(['a', 'b', 'c', 1.0]))
        assert ser['a'] == 1
        assert ser[1.0] == 1

    def test_getitem_float_keys_tuple_values(self):
        ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.1, 0.2], name='foo')
        result = ser[0.0]
        assert result == (1, 1)
        expected = Series([(1, 1), (2, 2)], index=[0.0, 0.0], name='foo')
        ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.0, 0.2], name='foo')
        result = ser[0.0]
        tm.assert_series_equal(result, expected)

    def test_getitem_unrecognized_scalar(self):
        ser = Series([1, 2], index=[np.dtype('O'), np.dtype('i8')])
        key = ser.index[1]
        result = ser[key]
        assert result == 2

    def test_getitem_negative_out_of_bounds(self):
        ser = Series(['a'] * 10, index=['a'] * 10)
        msg = 'index -11 is out of bounds for axis 0 with size 10|index out of bounds'
        warn_msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with pytest.raises(IndexError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                ser[-11]

    def test_getitem_out_of_bounds_indexerror(self, datetime_series):
        msg = 'index \\d+ is out of bounds for axis 0 with size \\d+'
        warn_msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with pytest.raises(IndexError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                datetime_series[len(datetime_series)]

    def test_getitem_out_of_bounds_empty_rangeindex_keyerror(self):
        ser = Series([], dtype=object)
        with pytest.raises(KeyError, match='-1'):
            ser[-1]

    def test_getitem_keyerror_with_integer_index(self, any_int_numpy_dtype):
        dtype = any_int_numpy_dtype
        ser = Series(np.random.default_rng(2).standard_normal(6), index=Index([0, 0, 1, 1, 2, 2], dtype=dtype))
        with pytest.raises(KeyError, match='^5$'):
            ser[5]
        with pytest.raises(KeyError, match="^'c'$"):
            ser['c']
        ser = Series(np.random.default_rng(2).standard_normal(6), index=[2, 2, 0, 0, 1, 1])
        with pytest.raises(KeyError, match='^5$'):
            ser[5]
        with pytest.raises(KeyError, match="^'c'$"):
            ser['c']

    def test_getitem_int64(self, datetime_series):
        idx = np.int64(5)
        msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = datetime_series[idx]
        assert res == datetime_series.iloc[5]

    def test_getitem_full_range(self):
        ser = Series(range(5), index=list(range(5)))
        result = ser[list(range(5))]
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize('tzstr', ['Europe/Berlin', 'dateutil/Europe/Berlin'])
    def test_getitem_pydatetime_tz(self, tzstr):
        tz = timezones.maybe_get_tz(tzstr)
        index = date_range(start='2012-12-24 16:00', end='2012-12-24 18:00', freq='h', tz=tzstr)
        ts = Series(index=index, data=index.hour)
        time_pandas = Timestamp('2012-12-24 17:00', tz=tzstr)
        dt = datetime(2012, 12, 24, 17, 0)
        time_datetime = conversion.localize_pydatetime(dt, tz)
        assert ts[time_pandas] == ts[time_datetime]

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_string_index_alias_tz_aware(self, tz):
        rng = date_range('1/1/2000', periods=10, tz=tz)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        result = ser['1/3/2000']
        tm.assert_almost_equal(result, ser.iloc[2])

    def test_getitem_time_object(self):
        rng = date_range('1/1/2000', '1/5/2000', freq='5min')
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        mask = (rng.hour == 9) & (rng.minute == 30)
        result = ts[time(9, 30)]
        expected = ts[mask]
        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    def test_getitem_scalar_categorical_index(self):
        cats = Categorical([Timestamp('12-31-1999'), Timestamp('12-31-2000')])
        ser = Series([1, 2], index=cats)
        expected = ser.iloc[0]
        result = ser[cats[0]]
        assert result == expected

    def test_getitem_numeric_categorical_listlike_matches_scalar(self):
        ser = Series(['a', 'b', 'c'], index=pd.CategoricalIndex([2, 1, 0]))
        assert ser[0] == 'c'
        res = ser[[0]]
        expected = ser.iloc[-1:]
        tm.assert_series_equal(res, expected)
        res2 = ser[[0, 1, 2]]
        tm.assert_series_equal(res2, ser.iloc[::-1])

    def test_getitem_integer_categorical_not_positional(self):
        ser = Series(['a', 'b', 'c'], index=Index([1, 2, 3], dtype='category'))
        assert ser.get(3) == 'c'
        assert ser[3] == 'c'

    def test_getitem_str_with_timedeltaindex(self):
        rng = timedelta_range('1 day 10:11:12', freq='h', periods=500)
        ser = Series(np.arange(len(rng)), index=rng)
        key = '6 days, 23:11:12'
        indexer = rng.get_loc(key)
        assert indexer == 133
        result = ser[key]
        assert result == ser.iloc[133]
        msg = "^Timedelta\\('50 days 00:00:00'\\)$"
        with pytest.raises(KeyError, match=msg):
            rng.get_loc('50 days')
        with pytest.raises(KeyError, match=msg):
            ser['50 days']

    def test_getitem_bool_index_positional(self):
        ser = Series({True: 1, False: 0})
        msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser[0]
        assert result == 1