import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestTimezoneConcat:

    def test_concat_tz_series(self):
        x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h', tz='UTC'))
        y = Series(date_range('2012-01-01', '2012-01-02'))
        expected = Series([x[0], x[1], y[0], y[1]], dtype='object')
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series2(self):
        x = Series(date_range('20151124 08:00', '20151124 09:00', freq='1h', tz='UTC'))
        y = Series(['a', 'b'])
        expected = Series([x[0], x[1], y[0], y[1]], dtype='object')
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series3(self, unit, unit2):
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f'M8[{unit}]')
        first[0] = first[0].dt.tz_localize('UTC')
        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f'M8[{unit2}]')
        second[0] = second[0].dt.tz_localize('UTC')
        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f'datetime64[{exp_unit}, UTC]'

    def test_concat_tz_series4(self, unit, unit2):
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f'M8[{unit}]')
        first[0] = first[0].dt.tz_localize('Europe/London')
        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f'M8[{unit2}]')
        second[0] = second[0].dt.tz_localize('Europe/London')
        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f'datetime64[{exp_unit}, Europe/London]'

    def test_concat_tz_series5(self, unit, unit2):
        first = DataFrame([[datetime(2016, 1, 1)], [datetime(2016, 1, 2)]], dtype=f'M8[{unit}]')
        first[0] = first[0].dt.tz_localize('Europe/London')
        second = DataFrame([[datetime(2016, 1, 3)]], dtype=f'M8[{unit2}]')
        second[0] = second[0].dt.tz_localize('Europe/London')
        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f'datetime64[{exp_unit}, Europe/London]'

    def test_concat_tz_series6(self, unit, unit2):
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f'M8[{unit}]')
        first[0] = first[0].dt.tz_localize('Europe/London')
        second = DataFrame([[datetime(2016, 1, 2)], [datetime(2016, 1, 3)]], dtype=f'M8[{unit2}]')
        second[0] = second[0].dt.tz_localize('Europe/London')
        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f'datetime64[{exp_unit}, Europe/London]'

    def test_concat_tz_series_tzlocal(self):
        x = [Timestamp('2011-01-01', tz=dateutil.tz.tzlocal()), Timestamp('2011-02-01', tz=dateutil.tz.tzlocal())]
        y = [Timestamp('2012-01-01', tz=dateutil.tz.tzlocal()), Timestamp('2012-02-01', tz=dateutil.tz.tzlocal())]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y))
        assert result.dtype == 'datetime64[ns, tzlocal()]'

    def test_concat_tz_series_with_datetimelike(self):
        x = [Timestamp('2011-01-01', tz='US/Eastern'), Timestamp('2011-02-01', tz='US/Eastern')]
        y = [pd.Timedelta('1 day'), pd.Timedelta('2 day')]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype='object'))
        y = [pd.Period('2011-03', freq='M'), pd.Period('2011-04', freq='M')]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype='object'))

    def test_concat_tz_frame(self):
        df2 = DataFrame({'A': Timestamp('20130102', tz='US/Eastern'), 'B': Timestamp('20130603', tz='CET')}, index=range(5))
        df3 = concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        tm.assert_frame_equal(df2, df3)

    def test_concat_multiple_tzs(self):
        ts1 = Timestamp('2015-01-01', tz=None)
        ts2 = Timestamp('2015-01-01', tz='UTC')
        ts3 = Timestamp('2015-01-01', tz='EST')
        df1 = DataFrame({'time': [ts1]})
        df2 = DataFrame({'time': [ts2]})
        df3 = DataFrame({'time': [ts3]})
        results = concat([df1, df2]).reset_index(drop=True)
        expected = DataFrame({'time': [ts1, ts2]}, dtype=object)
        tm.assert_frame_equal(results, expected)
        results = concat([df1, df3]).reset_index(drop=True)
        expected = DataFrame({'time': [ts1, ts3]}, dtype=object)
        tm.assert_frame_equal(results, expected)
        results = concat([df2, df3]).reset_index(drop=True)
        expected = DataFrame({'time': [ts2, ts3]})
        tm.assert_frame_equal(results, expected)

    def test_concat_multiindex_with_tz(self):
        df = DataFrame({'dt': DatetimeIndex([datetime(2014, 1, 1), datetime(2014, 1, 2), datetime(2014, 1, 3)], dtype='M8[ns, US/Pacific]'), 'b': ['A', 'B', 'C'], 'c': [1, 2, 3], 'd': [4, 5, 6]})
        df = df.set_index(['dt', 'b'])
        exp_idx1 = DatetimeIndex(['2014-01-01', '2014-01-02', '2014-01-03'] * 2, dtype='M8[ns, US/Pacific]', name='dt')
        exp_idx2 = Index(['A', 'B', 'C'] * 2, name='b')
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame({'c': [1, 2, 3] * 2, 'd': [4, 5, 6] * 2}, index=exp_idx, columns=['c', 'd'])
        result = concat([df, df])
        tm.assert_frame_equal(result, expected)

    def test_concat_tz_not_aligned(self):
        ts = pd.to_datetime([1, 2]).tz_localize('UTC')
        a = DataFrame({'A': ts})
        b = DataFrame({'A': ts, 'B': ts})
        result = concat([a, b], sort=True, ignore_index=True)
        expected = DataFrame({'A': list(ts) + list(ts), 'B': [pd.NaT, pd.NaT] + list(ts)})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('t1', ['2015-01-01', pytest.param(pd.NaT, marks=pytest.mark.xfail(reason='GH23037 incorrect dtype when concatenating'))])
    def test_concat_tz_NaT(self, t1):
        ts1 = Timestamp(t1, tz='UTC')
        ts2 = Timestamp('2015-01-01', tz='UTC')
        ts3 = Timestamp('2015-01-01', tz='UTC')
        df1 = DataFrame([[ts1, ts2]])
        df2 = DataFrame([[ts3]])
        result = concat([df1, df2])
        expected = DataFrame([[ts1, ts2], [ts3, pd.NaT]], index=[0, 0])
        tm.assert_frame_equal(result, expected)

    def test_concat_tz_with_empty(self):
        result = concat([DataFrame(date_range('2000', periods=1, tz='UTC')), DataFrame()])
        expected = DataFrame(date_range('2000', periods=1, tz='UTC'))
        tm.assert_frame_equal(result, expected)