import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
class TestPeriodIndexArithmetic:

    def test_parr_add_iadd_parr_raises(self, box_with_array):
        rng = period_range('1/1/2000', freq='D', periods=5)
        other = period_range('1/6/2000', freq='D', periods=5)
        rng = tm.box_expected(rng, box_with_array)
        msg = 'unsupported operand type\\(s\\) for \\+: .* and .*'
        with pytest.raises(TypeError, match=msg):
            rng + other
        with pytest.raises(TypeError, match=msg):
            rng += other

    def test_pi_sub_isub_pi(self):
        rng = period_range('1/1/2000', freq='D', periods=5)
        other = period_range('1/6/2000', freq='D', periods=5)
        off = rng.freq
        expected = pd.Index([-5 * off] * 5)
        result = rng - other
        tm.assert_index_equal(result, expected)
        rng -= other
        tm.assert_index_equal(rng, expected)

    def test_pi_sub_pi_with_nat(self):
        rng = period_range('1/1/2000', freq='D', periods=5)
        other = rng[1:].insert(0, pd.NaT)
        assert other[1:].equals(rng[1:])
        result = rng - other
        off = rng.freq
        expected = pd.Index([pd.NaT, 0 * off, 0 * off, 0 * off, 0 * off])
        tm.assert_index_equal(result, expected)

    def test_parr_sub_pi_mismatched_freq(self, box_with_array, box_with_array2):
        rng = period_range('1/1/2000', freq='D', periods=5)
        other = period_range('1/6/2000', freq='h', periods=5)
        rng = tm.box_expected(rng, box_with_array)
        other = tm.box_expected(other, box_with_array2)
        msg = 'Input has different freq=[hD] from PeriodArray\\(freq=[Dh]\\)'
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other

    @pytest.mark.parametrize('n', [1, 2, 3, 4])
    def test_sub_n_gt_1_ticks(self, tick_classes, n):
        p1_d = '19910905'
        p2_d = '19920406'
        p1 = PeriodIndex([p1_d], freq=tick_classes(n))
        p2 = PeriodIndex([p2_d], freq=tick_classes(n))
        expected = PeriodIndex([p2_d], freq=p2.freq.base) - PeriodIndex([p1_d], freq=p1.freq.base)
        tm.assert_index_equal(p2 - p1, expected)

    @pytest.mark.parametrize('n', [1, 2, 3, 4])
    @pytest.mark.parametrize('offset, kwd_name', [(pd.offsets.YearEnd, 'month'), (pd.offsets.QuarterEnd, 'startingMonth'), (pd.offsets.MonthEnd, None), (pd.offsets.Week, 'weekday')])
    def test_sub_n_gt_1_offsets(self, offset, kwd_name, n):
        kwds = {kwd_name: 3} if kwd_name is not None else {}
        p1_d = '19910905'
        p2_d = '19920406'
        freq = offset(n, normalize=False, **kwds)
        p1 = PeriodIndex([p1_d], freq=freq)
        p2 = PeriodIndex([p2_d], freq=freq)
        result = p2 - p1
        expected = PeriodIndex([p2_d], freq=freq.base) - PeriodIndex([p1_d], freq=freq.base)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('other', [Timestamp('2016-01-01'), Timestamp('2016-01-01').to_pydatetime(), Timestamp('2016-01-01').to_datetime64(), pd.date_range('2016-01-01', periods=3, freq='h'), pd.date_range('2016-01-01', periods=3, tz='Europe/Brussels'), pd.date_range('2016-01-01', periods=3, freq='s')._data, pd.date_range('2016-01-01', periods=3, tz='Asia/Tokyo')._data, 3.14, np.array([2.0, 3.0, 4.0])])
    def test_parr_add_sub_invalid(self, other, box_with_array):
        rng = period_range('1/1/2000', freq='D', periods=3)
        rng = tm.box_expected(rng, box_with_array)
        msg = '|'.join(['(:?cannot add PeriodArray and .*)', '(:?cannot subtract .* from (:?a\\s)?.*)', '(:?unsupported operand type\\(s\\) for \\+: .* and .*)', 'unsupported operand type\\(s\\) for [+-]: .* and .*'])
        assert_invalid_addsub_type(rng, other, msg)
        with pytest.raises(TypeError, match=msg):
            rng + other
        with pytest.raises(TypeError, match=msg):
            other + rng
        with pytest.raises(TypeError, match=msg):
            rng - other
        with pytest.raises(TypeError, match=msg):
            other - rng

    def test_pi_add_sub_td64_array_non_tick_raises(self):
        rng = period_range('1/1/2000', freq='Q', periods=3)
        tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
        tdarr = tdi.values
        msg = 'Cannot add or subtract timedelta64\\[ns\\] dtype from period\\[Q-DEC\\]'
        with pytest.raises(TypeError, match=msg):
            rng + tdarr
        with pytest.raises(TypeError, match=msg):
            tdarr + rng
        with pytest.raises(TypeError, match=msg):
            rng - tdarr
        msg = 'cannot subtract PeriodArray from TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdarr - rng

    def test_pi_add_sub_td64_array_tick(self):
        rng = period_range('1/1/2000', freq='90D', periods=3)
        tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
        tdarr = tdi.values
        expected = period_range('12/31/1999', freq='90D', periods=3)
        result = rng + tdi
        tm.assert_index_equal(result, expected)
        result = rng + tdarr
        tm.assert_index_equal(result, expected)
        result = tdi + rng
        tm.assert_index_equal(result, expected)
        result = tdarr + rng
        tm.assert_index_equal(result, expected)
        expected = period_range('1/2/2000', freq='90D', periods=3)
        result = rng - tdi
        tm.assert_index_equal(result, expected)
        result = rng - tdarr
        tm.assert_index_equal(result, expected)
        msg = 'cannot subtract .* from .*'
        with pytest.raises(TypeError, match=msg):
            tdarr - rng
        with pytest.raises(TypeError, match=msg):
            tdi - rng

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('tdi_freq', [None, 'h'])
    def test_parr_sub_td64array(self, box_with_array, tdi_freq, pi_freq):
        box = box_with_array
        xbox = box if box not in [pd.array, tm.to_array] else pd.Index
        tdi = TimedeltaIndex(['1 hours', '2 hours'], freq=tdi_freq)
        dti = Timestamp('2018-03-07 17:16:40') + tdi
        pi = dti.to_period(pi_freq)
        td64obj = tm.box_expected(tdi, box)
        if pi_freq == 'h':
            result = pi - td64obj
            expected = (pi.to_timestamp('s') - tdi).to_period(pi_freq)
            expected = tm.box_expected(expected, xbox)
            tm.assert_equal(result, expected)
            result = pi[0] - td64obj
            expected = (pi[0].to_timestamp('s') - tdi).to_period(pi_freq)
            expected = tm.box_expected(expected, box)
            tm.assert_equal(result, expected)
        elif pi_freq == 'D':
            msg = "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."
            with pytest.raises(IncompatibleFrequency, match=msg):
                pi - td64obj
            with pytest.raises(IncompatibleFrequency, match=msg):
                pi[0] - td64obj
        else:
            msg = 'Cannot add or subtract timedelta64'
            with pytest.raises(TypeError, match=msg):
                pi - td64obj
            with pytest.raises(TypeError, match=msg):
                pi[0] - td64obj

    @pytest.mark.parametrize('box', [np.array, pd.Index])
    def test_pi_add_offset_array(self, box):
        pi = PeriodIndex([Period('2015Q1'), Period('2016Q2')])
        offs = box([pd.offsets.QuarterEnd(n=1, startingMonth=12), pd.offsets.QuarterEnd(n=-2, startingMonth=12)])
        expected = PeriodIndex([Period('2015Q2'), Period('2015Q4')]).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res = pi + offs
        tm.assert_index_equal(res, expected)
        with tm.assert_produces_warning(PerformanceWarning):
            res2 = offs + pi
        tm.assert_index_equal(res2, expected)
        unanchored = np.array([pd.offsets.Hour(n=1), pd.offsets.Minute(n=-2)])
        msg = 'Input cannot be converted to Period\\(freq=Q-DEC\\)'
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                pi + unanchored
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                unanchored + pi

    @pytest.mark.parametrize('box', [np.array, pd.Index])
    def test_pi_sub_offset_array(self, box):
        pi = PeriodIndex([Period('2015Q1'), Period('2016Q2')])
        other = box([pd.offsets.QuarterEnd(n=1, startingMonth=12), pd.offsets.QuarterEnd(n=-2, startingMonth=12)])
        expected = PeriodIndex([pi[n] - other[n] for n in range(len(pi))])
        expected = expected.astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res = pi - other
        tm.assert_index_equal(res, expected)
        anchored = box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)])
        msg = 'Input has different freq=-1M from Period\\(freq=Q-DEC\\)'
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                pi - anchored
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored - pi

    def test_pi_add_iadd_int(self, one):
        rng = period_range('2000-01-01 09:00', freq='h', periods=10)
        result = rng + one
        expected = period_range('2000-01-01 10:00', freq='h', periods=10)
        tm.assert_index_equal(result, expected)
        rng += one
        tm.assert_index_equal(rng, expected)

    def test_pi_sub_isub_int(self, one):
        """
        PeriodIndex.__sub__ and __isub__ with several representations of
        the integer 1, e.g. int, np.int64, np.uint8, ...
        """
        rng = period_range('2000-01-01 09:00', freq='h', periods=10)
        result = rng - one
        expected = period_range('2000-01-01 08:00', freq='h', periods=10)
        tm.assert_index_equal(result, expected)
        rng -= one
        tm.assert_index_equal(rng, expected)

    @pytest.mark.parametrize('five', [5, np.array(5, dtype=np.int64)])
    def test_pi_sub_intlike(self, five):
        rng = period_range('2007-01', periods=50)
        result = rng - five
        exp = rng + -five
        tm.assert_index_equal(result, exp)

    def test_pi_add_sub_int_array_freqn_gt1(self):
        pi = period_range('2016-01-01', periods=10, freq='2D')
        arr = np.arange(10)
        result = pi + arr
        expected = pd.Index([x + y for x, y in zip(pi, arr)])
        tm.assert_index_equal(result, expected)
        result = pi - arr
        expected = pd.Index([x - y for x, y in zip(pi, arr)])
        tm.assert_index_equal(result, expected)

    def test_pi_sub_isub_offset(self):
        rng = period_range('2014', '2024', freq='Y')
        result = rng - pd.offsets.YearEnd(5)
        expected = period_range('2009', '2019', freq='Y')
        tm.assert_index_equal(result, expected)
        rng -= pd.offsets.YearEnd(5)
        tm.assert_index_equal(rng, expected)
        rng = period_range('2014-01', '2016-12', freq='M')
        result = rng - pd.offsets.MonthEnd(5)
        expected = period_range('2013-08', '2016-07', freq='M')
        tm.assert_index_equal(result, expected)
        rng -= pd.offsets.MonthEnd(5)
        tm.assert_index_equal(rng, expected)

    @pytest.mark.parametrize('transpose', [True, False])
    def test_pi_add_offset_n_gt1(self, box_with_array, transpose):
        per = Period('2016-01', freq='2M')
        pi = PeriodIndex([per])
        expected = PeriodIndex(['2016-03'], freq='2M')
        pi = tm.box_expected(pi, box_with_array, transpose=transpose)
        expected = tm.box_expected(expected, box_with_array, transpose=transpose)
        result = pi + per.freq
        tm.assert_equal(result, expected)
        result = per.freq + pi
        tm.assert_equal(result, expected)

    def test_pi_add_offset_n_gt1_not_divisible(self, box_with_array):
        pi = PeriodIndex(['2016-01'], freq='2M')
        expected = PeriodIndex(['2016-04'], freq='2M')
        pi = tm.box_expected(pi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = pi + to_offset('3ME')
        tm.assert_equal(result, expected)
        result = to_offset('3ME') + pi
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('int_holder', [np.array, pd.Index])
    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_pi_add_intarray(self, int_holder, op):
        pi = PeriodIndex([Period('2015Q1'), Period('NaT')])
        other = int_holder([4, -1])
        result = op(pi, other)
        expected = PeriodIndex([Period('2016Q1'), Period('NaT')])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('int_holder', [np.array, pd.Index])
    def test_pi_sub_intarray(self, int_holder):
        pi = PeriodIndex([Period('2015Q1'), Period('NaT')])
        other = int_holder([4, -1])
        result = pi - other
        expected = PeriodIndex([Period('2014Q1'), Period('NaT')])
        tm.assert_index_equal(result, expected)
        msg = "bad operand type for unary -: 'PeriodArray'"
        with pytest.raises(TypeError, match=msg):
            other - pi

    def test_parr_add_timedeltalike_minute_gt1(self, three_days, box_with_array):
        other = three_days
        rng = period_range('2014-05-01', periods=3, freq='2D')
        rng = tm.box_expected(rng, box_with_array)
        expected = PeriodIndex(['2014-05-04', '2014-05-06', '2014-05-08'], freq='2D')
        expected = tm.box_expected(expected, box_with_array)
        result = rng + other
        tm.assert_equal(result, expected)
        result = other + rng
        tm.assert_equal(result, expected)
        expected = PeriodIndex(['2014-04-28', '2014-04-30', '2014-05-02'], freq='2D')
        expected = tm.box_expected(expected, box_with_array)
        result = rng - other
        tm.assert_equal(result, expected)
        msg = '|'.join(["bad operand type for unary -: 'PeriodArray'", 'cannot subtract PeriodArray from timedelta64\\[[hD]\\]'])
        with pytest.raises(TypeError, match=msg):
            other - rng

    @pytest.mark.parametrize('freqstr', ['5ns', '5us', '5ms', '5s', '5min', '5h', '5d'])
    def test_parr_add_timedeltalike_tick_gt1(self, three_days, freqstr, box_with_array):
        other = three_days
        rng = period_range('2014-05-01', periods=6, freq=freqstr)
        first = rng[0]
        rng = tm.box_expected(rng, box_with_array)
        expected = period_range(first + other, periods=6, freq=freqstr)
        expected = tm.box_expected(expected, box_with_array)
        result = rng + other
        tm.assert_equal(result, expected)
        result = other + rng
        tm.assert_equal(result, expected)
        expected = period_range(first - other, periods=6, freq=freqstr)
        expected = tm.box_expected(expected, box_with_array)
        result = rng - other
        tm.assert_equal(result, expected)
        msg = '|'.join(["bad operand type for unary -: 'PeriodArray'", 'cannot subtract PeriodArray from timedelta64\\[[hD]\\]'])
        with pytest.raises(TypeError, match=msg):
            other - rng

    def test_pi_add_iadd_timedeltalike_daily(self, three_days):
        other = three_days
        rng = period_range('2014-05-01', '2014-05-15', freq='D')
        expected = period_range('2014-05-04', '2014-05-18', freq='D')
        result = rng + other
        tm.assert_index_equal(result, expected)
        rng += other
        tm.assert_index_equal(rng, expected)

    def test_pi_sub_isub_timedeltalike_daily(self, three_days):
        other = three_days
        rng = period_range('2014-05-01', '2014-05-15', freq='D')
        expected = period_range('2014-04-28', '2014-05-12', freq='D')
        result = rng - other
        tm.assert_index_equal(result, expected)
        rng -= other
        tm.assert_index_equal(rng, expected)

    def test_parr_add_sub_timedeltalike_freq_mismatch_daily(self, not_daily, box_with_array):
        other = not_daily
        rng = period_range('2014-05-01', '2014-05-15', freq='D')
        rng = tm.box_expected(rng, box_with_array)
        msg = '|'.join(['Input has different freq(=.+)? from Period.*?\\(freq=D\\)', "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."])
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng -= other

    def test_pi_add_iadd_timedeltalike_hourly(self, two_hours):
        other = two_hours
        rng = period_range('2014-01-01 10:00', '2014-01-05 10:00', freq='h')
        expected = period_range('2014-01-01 12:00', '2014-01-05 12:00', freq='h')
        result = rng + other
        tm.assert_index_equal(result, expected)
        rng += other
        tm.assert_index_equal(rng, expected)

    def test_parr_add_timedeltalike_mismatched_freq_hourly(self, not_hourly, box_with_array):
        other = not_hourly
        rng = period_range('2014-01-01 10:00', '2014-01-05 10:00', freq='h')
        rng = tm.box_expected(rng, box_with_array)
        msg = '|'.join(['Input has different freq(=.+)? from Period.*?\\(freq=h\\)', "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."])
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other

    def test_pi_sub_isub_timedeltalike_hourly(self, two_hours):
        other = two_hours
        rng = period_range('2014-01-01 10:00', '2014-01-05 10:00', freq='h')
        expected = period_range('2014-01-01 08:00', '2014-01-05 08:00', freq='h')
        result = rng - other
        tm.assert_index_equal(result, expected)
        rng -= other
        tm.assert_index_equal(rng, expected)

    def test_add_iadd_timedeltalike_annual(self):
        rng = period_range('2014', '2024', freq='Y')
        result = rng + pd.offsets.YearEnd(5)
        expected = period_range('2019', '2029', freq='Y')
        tm.assert_index_equal(result, expected)
        rng += pd.offsets.YearEnd(5)
        tm.assert_index_equal(rng, expected)

    def test_pi_add_sub_timedeltalike_freq_mismatch_annual(self, mismatched_freq):
        other = mismatched_freq
        rng = period_range('2014', '2024', freq='Y')
        msg = 'Input has different freq(=.+)? from Period.*?\\(freq=Y-DEC\\)'
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng -= other

    def test_pi_add_iadd_timedeltalike_M(self):
        rng = period_range('2014-01', '2016-12', freq='M')
        expected = period_range('2014-06', '2017-05', freq='M')
        result = rng + pd.offsets.MonthEnd(5)
        tm.assert_index_equal(result, expected)
        rng += pd.offsets.MonthEnd(5)
        tm.assert_index_equal(rng, expected)

    def test_pi_add_sub_timedeltalike_freq_mismatch_monthly(self, mismatched_freq):
        other = mismatched_freq
        rng = period_range('2014-01', '2016-12', freq='M')
        msg = 'Input has different freq(=.+)? from Period.*?\\(freq=M\\)'
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng -= other

    @pytest.mark.parametrize('transpose', [True, False])
    def test_parr_add_sub_td64_nat(self, box_with_array, transpose):
        pi = period_range('1994-04-01', periods=9, freq='19D')
        other = np.timedelta64('NaT')
        expected = PeriodIndex(['NaT'] * 9, freq='19D')
        obj = tm.box_expected(pi, box_with_array, transpose=transpose)
        expected = tm.box_expected(expected, box_with_array, transpose=transpose)
        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg = 'cannot subtract .* from .*'
        with pytest.raises(TypeError, match=msg):
            other - obj

    @pytest.mark.parametrize('other', [np.array(['NaT'] * 9, dtype='m8[ns]'), TimedeltaArray._from_sequence(['NaT'] * 9, dtype='m8[ns]')])
    def test_parr_add_sub_tdt64_nat_array(self, box_with_array, other):
        pi = period_range('1994-04-01', periods=9, freq='19D')
        expected = PeriodIndex(['NaT'] * 9, freq='19D')
        obj = tm.box_expected(pi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg = 'cannot subtract .* from .*'
        with pytest.raises(TypeError, match=msg):
            other - obj
        other = other.copy()
        other[0] = np.timedelta64(0, 'ns')
        expected = PeriodIndex([pi[0]] + ['NaT'] * 8, freq='19D')
        expected = tm.box_expected(expected, box_with_array)
        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_parr_add_sub_index(self):
        pi = period_range('2000-12-31', periods=3)
        parr = pi.array
        result = parr - pi
        expected = pi - pi
        tm.assert_index_equal(result, expected)

    def test_parr_add_sub_object_array(self):
        pi = period_range('2000-12-31', periods=3, freq='D')
        parr = pi.array
        other = np.array([Timedelta(days=1), pd.offsets.Day(2), 3])
        with tm.assert_produces_warning(PerformanceWarning):
            result = parr + other
        expected = PeriodIndex(['2001-01-01', '2001-01-03', '2001-01-05'], freq='D')._data.astype(object)
        tm.assert_equal(result, expected)
        with tm.assert_produces_warning(PerformanceWarning):
            result = parr - other
        expected = PeriodIndex(['2000-12-30'] * 3, freq='D')._data.astype(object)
        tm.assert_equal(result, expected)

    def test_period_add_timestamp_raises(self, box_with_array):
        ts = Timestamp('2017')
        per = Period('2017', freq='M')
        arr = pd.Index([per], dtype='Period[M]')
        arr = tm.box_expected(arr, box_with_array)
        msg = 'cannot add PeriodArray and Timestamp'
        with pytest.raises(TypeError, match=msg):
            arr + ts
        with pytest.raises(TypeError, match=msg):
            ts + arr
        msg = 'cannot add PeriodArray and DatetimeArray'
        with pytest.raises(TypeError, match=msg):
            arr + Series([ts])
        with pytest.raises(TypeError, match=msg):
            Series([ts]) + arr
        with pytest.raises(TypeError, match=msg):
            arr + pd.Index([ts])
        with pytest.raises(TypeError, match=msg):
            pd.Index([ts]) + arr
        if box_with_array is pd.DataFrame:
            msg = 'cannot add PeriodArray and DatetimeArray'
        else:
            msg = "unsupported operand type\\(s\\) for \\+: 'Period' and 'DatetimeArray"
        with pytest.raises(TypeError, match=msg):
            arr + pd.DataFrame([ts])
        if box_with_array is pd.DataFrame:
            msg = 'cannot add PeriodArray and DatetimeArray'
        else:
            msg = "unsupported operand type\\(s\\) for \\+: 'DatetimeArray' and 'Period'"
        with pytest.raises(TypeError, match=msg):
            pd.DataFrame([ts]) + arr