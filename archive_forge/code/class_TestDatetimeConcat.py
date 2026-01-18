import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDatetimeConcat:

    def test_concat_datetime64_block(self):
        rng = date_range('1/1/2000', periods=10)
        df = DataFrame({'time': rng})
        result = concat([df, df])
        assert (result.iloc[:10]['time'] == rng).all()
        assert (result.iloc[10:]['time'] == rng).all()

    def test_concat_datetime_datetime64_frame(self):
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), 'hi'])
        df2_obj = DataFrame.from_records(rows, columns=['date', 'test'])
        ind = date_range(start='2000/1/1', freq='D', periods=10)
        df1 = DataFrame({'date': ind, 'test': range(10)})
        concat([df1, df2_obj])

    def test_concat_datetime_timezone(self):
        idx1 = date_range('2011-01-01', periods=3, freq='h', tz='Europe/Paris')
        idx2 = date_range(start=idx1[0], end=idx1[-1], freq='h')
        df1 = DataFrame({'a': [1, 2, 3]}, index=idx1)
        df2 = DataFrame({'b': [1, 2, 3]}, index=idx2)
        result = concat([df1, df2], axis=1)
        exp_idx = DatetimeIndex(['2011-01-01 00:00:00+01:00', '2011-01-01 01:00:00+01:00', '2011-01-01 02:00:00+01:00'], dtype='M8[ns, Europe/Paris]', freq='h')
        expected = DataFrame([[1, 1], [2, 2], [3, 3]], index=exp_idx, columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)
        idx3 = date_range('2011-01-01', periods=3, freq='h', tz='Asia/Tokyo')
        df3 = DataFrame({'b': [1, 2, 3]}, index=idx3)
        result = concat([df1, df3], axis=1)
        exp_idx = DatetimeIndex(['2010-12-31 15:00:00+00:00', '2010-12-31 16:00:00+00:00', '2010-12-31 17:00:00+00:00', '2010-12-31 23:00:00+00:00', '2011-01-01 00:00:00+00:00', '2011-01-01 01:00:00+00:00']).as_unit('ns')
        expected = DataFrame([[np.nan, 1], [np.nan, 2], [np.nan, 3], [1, np.nan], [2, np.nan], [3, np.nan]], index=exp_idx, columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)
        result = concat([df1.resample('h').mean(), df2.resample('h').mean()], sort=True)
        expected = DataFrame({'a': [1, 2, 3] + [np.nan] * 3, 'b': [np.nan] * 3 + [1, 2, 3]}, index=idx1.append(idx1))
        tm.assert_frame_equal(result, expected)

    def test_concat_datetimeindex_freq(self):
        dr = date_range('01-Jan-2013', periods=100, freq='50ms', tz='UTC')
        data = list(range(100))
        expected = DataFrame(data, index=dr)
        result = concat([expected[:50], expected[50:]])
        tm.assert_frame_equal(result, expected)
        result = concat([expected[50:], expected[:50]])
        expected = DataFrame(data[50:] + data[:50], index=dr[50:].append(dr[:50]))
        expected.index._data.freq = None
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_datetime_object_index(self):
        idx = Index([dt.date(2013, 1, 1), dt.date(2014, 1, 1), dt.date(2015, 1, 1)], dtype='object')
        s = Series(['a', 'b'], index=MultiIndex.from_arrays([[1, 2], idx[:-1]], names=['first', 'second']))
        s2 = Series(['a', 'b'], index=MultiIndex.from_arrays([[1, 2], idx[::2]], names=['first', 'second']))
        mi = MultiIndex.from_arrays([[1, 2, 2], idx], names=['first', 'second'])
        assert mi.levels[1].dtype == object
        expected = DataFrame([['a', 'a'], ['b', np.nan], [np.nan, 'b']], index=mi)
        result = concat([s, s2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_NaT_series(self):
        x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h', tz='US/Eastern'))
        y = Series(pd.NaT, index=[0, 1], dtype='datetime64[ns, US/Eastern]')
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        expected = Series(pd.NaT, index=range(4), dtype='datetime64[ns, US/Eastern]')
        result = concat([y, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_NaT_series2(self):
        x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h'))
        y = Series(date_range('20151124 10:00', '20151124 11:00', freq='1h'))
        y[:] = pd.NaT
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        x[:] = pd.NaT
        expected = Series(pd.NaT, index=range(4), dtype='datetime64[ns]')
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_concat_NaT_dataframes(self, tz):
        dti = DatetimeIndex([pd.NaT, pd.NaT], tz=tz)
        first = DataFrame({0: dti})
        second = DataFrame([[Timestamp('2015/01/01', tz=tz)], [Timestamp('2016/01/01', tz=tz)]], index=[2, 3])
        expected = DataFrame([pd.NaT, pd.NaT, Timestamp('2015/01/01', tz=tz), Timestamp('2016/01/01', tz=tz)])
        result = concat([first, second], axis=0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz1', [None, 'UTC'])
    @pytest.mark.parametrize('tz2', [None, 'UTC'])
    @pytest.mark.parametrize('item', [pd.NaT, Timestamp('20150101')])
    def test_concat_NaT_dataframes_all_NaT_axis_0(self, tz1, tz2, item, using_array_manager):
        first = DataFrame([[pd.NaT], [pd.NaT]]).apply(lambda x: x.dt.tz_localize(tz1))
        second = DataFrame([item]).apply(lambda x: x.dt.tz_localize(tz2))
        result = concat([first, second], axis=0)
        expected = DataFrame(Series([pd.NaT, pd.NaT, item], index=[0, 1, 0]))
        expected = expected.apply(lambda x: x.dt.tz_localize(tz2))
        if tz1 != tz2:
            expected = expected.astype(object)
            if item is pd.NaT and (not using_array_manager):
                if tz1 is not None:
                    expected.iloc[-1, 0] = np.nan
                else:
                    expected.iloc[:-1, 0] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz1', [None, 'UTC'])
    @pytest.mark.parametrize('tz2', [None, 'UTC'])
    def test_concat_NaT_dataframes_all_NaT_axis_1(self, tz1, tz2):
        first = DataFrame(Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1))
        second = DataFrame(Series([pd.NaT]).dt.tz_localize(tz2), columns=[1])
        expected = DataFrame({0: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1), 1: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz2)})
        result = concat([first, second], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz1', [None, 'UTC'])
    @pytest.mark.parametrize('tz2', [None, 'UTC'])
    def test_concat_NaT_series_dataframe_all_NaT(self, tz1, tz2):
        first = Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1)
        second = DataFrame([[Timestamp('2015/01/01', tz=tz2)], [Timestamp('2016/01/01', tz=tz2)]], index=[2, 3])
        expected = DataFrame([pd.NaT, pd.NaT, Timestamp('2015/01/01', tz=tz2), Timestamp('2016/01/01', tz=tz2)])
        if tz1 != tz2:
            expected = expected.astype(object)
        result = concat([first, second])
        tm.assert_frame_equal(result, expected)