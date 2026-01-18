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
class TestDatetime64DateOffsetArithmetic:

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array, unit):
        ser = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')]).dt.as_unit(unit)
        expected = Series([Timestamp('20130101 9:01:05'), Timestamp('20130101 9:02:05')]).dt.as_unit(unit)
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected)
        result2 = pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)

    def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array):
        ser = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        expected = Series([Timestamp('20130101 9:00:55'), Timestamp('20130101 9:01:55')])
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = ser - pd.offsets.Second(5)
        tm.assert_equal(result, expected)
        result2 = -pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)
        msg = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            pd.offsets.Second(5) - ser

    @pytest.mark.parametrize('cls_name', ['Day', 'Hour', 'Minute', 'Second', 'Milli', 'Micro', 'Nano'])
    def test_dt64arr_add_sub_tick_DateOffset_smoke(self, cls_name, box_with_array):
        ser = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        ser = tm.box_expected(ser, box_with_array)
        offset_cls = getattr(pd.offsets, cls_name)
        ser + offset_cls(5)
        offset_cls(5) + ser
        ser - offset_cls(5)

    def test_dti_add_tick_tzaware(self, tz_aware_fixture, box_with_array):
        tz = tz_aware_fixture
        if tz == 'US/Pacific':
            dates = date_range('2012-11-01', periods=3, tz=tz)
            offset = dates + pd.offsets.Hour(5)
            assert dates[0] + pd.offsets.Hour(5) == offset[0]
        dates = date_range('2010-11-01 00:00', periods=3, tz=tz, freq='h')
        expected = DatetimeIndex(['2010-11-01 05:00', '2010-11-01 06:00', '2010-11-01 07:00'], freq='h', tz=tz).as_unit('ns')
        dates = tm.box_expected(dates, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        for scalar in [pd.offsets.Hour(5), np.timedelta64(5, 'h'), timedelta(hours=5)]:
            offset = dates + scalar
            tm.assert_equal(offset, expected)
            offset = scalar + dates
            tm.assert_equal(offset, expected)
            roundtrip = offset - scalar
            tm.assert_equal(roundtrip, dates)
            msg = '|'.join(['bad operand type for unary -', 'cannot subtract DatetimeArray'])
            with pytest.raises(TypeError, match=msg):
                scalar - dates

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array, unit):
        vec = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'), Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')]).as_unit(unit)
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec
        relative_kwargs = [('years', 2), ('months', 5), ('days', 3), ('hours', 5), ('minutes', 10), ('seconds', 2), ('microseconds', 5)]
        for i, (offset_unit, value) in enumerate(relative_kwargs):
            off = DateOffset(**{offset_unit: value})
            exp_unit = unit
            if offset_unit == 'microseconds' and unit != 'ns':
                exp_unit = 'us'
            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)
            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            off = DateOffset(**dict(relative_kwargs[:i + 1]))
            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)
            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            msg = '(bad|unsupported) operand type for unary'
            with pytest.raises(TypeError, match=msg):
                off - vec

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    @pytest.mark.parametrize('cls_and_kwargs', ['YearBegin', ('YearBegin', {'month': 5}), 'YearEnd', ('YearEnd', {'month': 5}), 'MonthBegin', 'MonthEnd', 'SemiMonthEnd', 'SemiMonthBegin', 'Week', ('Week', {'weekday': 3}), 'Week', ('Week', {'weekday': 6}), 'BusinessDay', 'BDay', 'QuarterEnd', 'QuarterBegin', 'CustomBusinessDay', 'CDay', 'CBMonthEnd', 'CBMonthBegin', 'BMonthBegin', 'BMonthEnd', 'BusinessHour', 'BYearBegin', 'BYearEnd', 'BQuarterBegin', ('LastWeekOfMonth', {'weekday': 2}), ('FY5253Quarter', {'qtr_with_extra_week': 1, 'startingMonth': 1, 'weekday': 2, 'variation': 'nearest'}), ('FY5253', {'weekday': 0, 'startingMonth': 2, 'variation': 'nearest'}), ('WeekOfMonth', {'weekday': 2, 'week': 2}), 'Easter', ('DateOffset', {'day': 4}), ('DateOffset', {'month': 5})])
    @pytest.mark.parametrize('normalize', [True, False])
    @pytest.mark.parametrize('n', [0, 5])
    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_dt64arr_add_sub_DateOffsets(self, box_with_array, n, normalize, cls_and_kwargs, unit, tz):
        if isinstance(cls_and_kwargs, tuple):
            cls_name, kwargs = cls_and_kwargs
        else:
            cls_name = cls_and_kwargs
            kwargs = {}
        if n == 0 and cls_name in ['WeekOfMonth', 'LastWeekOfMonth', 'FY5253Quarter', 'FY5253']:
            return
        vec = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'), Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')]).as_unit(unit).tz_localize(tz)
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec
        offset_cls = getattr(pd.offsets, cls_name)
        offset = offset_cls(n, normalize=normalize, **kwargs)
        expected = DatetimeIndex([x + offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec + offset)
        tm.assert_equal(expected, offset + vec)
        expected = DatetimeIndex([x - offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec - offset)
        expected = DatetimeIndex([offset + x for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, offset + vec)
        msg = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            offset - vec

    @pytest.mark.parametrize('other', [np.array([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)]), np.array([pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()]), np.array([pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)])])
    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dt64arr_add_sub_offset_array(self, tz_naive_fixture, box_with_array, op, other):
        tz = tz_naive_fixture
        dti = date_range('2017-01-01', periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        expected = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
        expected = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res = op(dtarr, other)
        tm.assert_equal(res, expected)
        other = tm.box_expected(other, box_with_array)
        if box_with_array is pd.array and op is roperator.radd:
            expected = pd.array(expected, dtype=object)
        with tm.assert_produces_warning(PerformanceWarning):
            res = op(dtarr, other)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('op, offset, exp, exp_freq', [('__add__', DateOffset(months=3, days=10), [Timestamp('2014-04-11'), Timestamp('2015-04-11'), Timestamp('2016-04-11'), Timestamp('2017-04-11')], None), ('__add__', DateOffset(months=3), [Timestamp('2014-04-01'), Timestamp('2015-04-01'), Timestamp('2016-04-01'), Timestamp('2017-04-01')], 'YS-APR'), ('__sub__', DateOffset(months=3, days=10), [Timestamp('2013-09-21'), Timestamp('2014-09-21'), Timestamp('2015-09-21'), Timestamp('2016-09-21')], None), ('__sub__', DateOffset(months=3), [Timestamp('2013-10-01'), Timestamp('2014-10-01'), Timestamp('2015-10-01'), Timestamp('2016-10-01')], 'YS-OCT')])
    def test_dti_add_sub_nonzero_mth_offset(self, op, offset, exp, exp_freq, tz_aware_fixture, box_with_array):
        tz = tz_aware_fixture
        date = date_range(start='01 Jan 2014', end='01 Jan 2017', freq='YS', tz=tz)
        date = tm.box_expected(date, box_with_array, False)
        mth = getattr(date, op)
        result = mth(offset)
        expected = DatetimeIndex(exp, tz=tz).as_unit('ns')
        expected = tm.box_expected(expected, box_with_array, False)
        tm.assert_equal(result, expected)

    def test_dt64arr_series_add_DateOffset_with_milli(self):
        dti = DatetimeIndex(['2000-01-01 00:00:00.012345678', '2000-01-31 00:00:00.012345678', '2000-02-29 00:00:00.012345678'], dtype='datetime64[ns]')
        result = dti + DateOffset(milliseconds=4)
        expected = DatetimeIndex(['2000-01-01 00:00:00.016345678', '2000-01-31 00:00:00.016345678', '2000-02-29 00:00:00.016345678'], dtype='datetime64[ns]')
        tm.assert_index_equal(result, expected)
        result = dti + DateOffset(days=1, milliseconds=4)
        expected = DatetimeIndex(['2000-01-02 00:00:00.016345678', '2000-02-01 00:00:00.016345678', '2000-03-01 00:00:00.016345678'], dtype='datetime64[ns]')
        tm.assert_index_equal(result, expected)