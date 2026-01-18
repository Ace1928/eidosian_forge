from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
class TestTimedeltaAdditionSubtraction:
    """
    Tests for Timedelta methods:

        __add__, __radd__,
        __sub__, __rsub__
    """

    @pytest.mark.parametrize('ten_seconds', [Timedelta(10, unit='s'), timedelta(seconds=10), np.timedelta64(10, 's'), np.timedelta64(10000000000, 'ns'), offsets.Second(10)])
    def test_td_add_sub_ten_seconds(self, ten_seconds):
        base = Timestamp('20130101 09:01:12.123456')
        expected_add = Timestamp('20130101 09:01:22.123456')
        expected_sub = Timestamp('20130101 09:01:02.123456')
        result = base + ten_seconds
        assert result == expected_add
        result = base - ten_seconds
        assert result == expected_sub

    @pytest.mark.parametrize('one_day_ten_secs', [Timedelta('1 day, 00:00:10'), Timedelta('1 days, 00:00:10'), timedelta(days=1, seconds=10), np.timedelta64(1, 'D') + np.timedelta64(10, 's'), offsets.Day() + offsets.Second(10)])
    def test_td_add_sub_one_day_ten_seconds(self, one_day_ten_secs):
        base = Timestamp('20130102 09:01:12.123456')
        expected_add = Timestamp('20130103 09:01:22.123456')
        expected_sub = Timestamp('20130101 09:01:02.123456')
        result = base + one_day_ten_secs
        assert result == expected_add
        result = base - one_day_ten_secs
        assert result == expected_sub

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_datetimelike_scalar(self, op):
        td = Timedelta(10, unit='d')
        result = op(td, datetime(2016, 1, 1))
        if op is operator.add:
            assert isinstance(result, Timestamp)
        assert result == Timestamp(2016, 1, 11)
        result = op(td, Timestamp('2018-01-12 18:09'))
        assert isinstance(result, Timestamp)
        assert result == Timestamp('2018-01-22 18:09')
        result = op(td, np.datetime64('2018-01-12'))
        assert isinstance(result, Timestamp)
        assert result == Timestamp('2018-01-22')
        result = op(td, NaT)
        assert result is NaT

    def test_td_add_timestamp_overflow(self):
        ts = Timestamp('1700-01-01').as_unit('ns')
        msg = "Cannot cast 259987 from D to 'ns' without overflow."
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            ts + Timedelta(13 * 19999, unit='D')
        msg = "Cannot cast 259987 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            ts + timedelta(days=13 * 19999)

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_td(self, op):
        td = Timedelta(10, unit='d')
        result = op(td, Timedelta(days=10))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=20)

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_pytimedelta(self, op):
        td = Timedelta(10, unit='d')
        result = op(td, timedelta(days=9))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=19)

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_timedelta64(self, op):
        td = Timedelta(10, unit='d')
        result = op(td, np.timedelta64(-4, 'D'))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=6)

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_offset(self, op):
        td = Timedelta(10, unit='d')
        result = op(td, offsets.Hour(6))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=10, hours=6)

    def test_td_sub_td(self):
        td = Timedelta(10, unit='d')
        expected = Timedelta(0, unit='ns')
        result = td - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_pytimedelta(self):
        td = Timedelta(10, unit='d')
        expected = Timedelta(0, unit='ns')
        result = td - td.to_pytimedelta()
        assert isinstance(result, Timedelta)
        assert result == expected
        result = td.to_pytimedelta() - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_timedelta64(self):
        td = Timedelta(10, unit='d')
        expected = Timedelta(0, unit='ns')
        result = td - td.to_timedelta64()
        assert isinstance(result, Timedelta)
        assert result == expected
        result = td.to_timedelta64() - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_nat(self):
        td = Timedelta(10, unit='d')
        result = td - NaT
        assert result is NaT

    def test_td_sub_td64_nat(self):
        td = Timedelta(10, unit='d')
        td_nat = np.timedelta64('NaT')
        result = td - td_nat
        assert result is NaT
        result = td_nat - td
        assert result is NaT

    def test_td_sub_offset(self):
        td = Timedelta(10, unit='d')
        result = td - offsets.Hour(1)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(239, unit='h')

    def test_td_add_sub_numeric_raises(self):
        td = Timedelta(10, unit='d')
        msg = 'unsupported operand type'
        for other in [2, 2.0, np.int64(2), np.float64(2)]:
            with pytest.raises(TypeError, match=msg):
                td + other
            with pytest.raises(TypeError, match=msg):
                other + td
            with pytest.raises(TypeError, match=msg):
                td - other
            with pytest.raises(TypeError, match=msg):
                other - td

    def test_td_add_sub_int_ndarray(self):
        td = Timedelta('1 day')
        other = np.array([1])
        msg = "unsupported operand type\\(s\\) for \\+: 'Timedelta' and 'int'"
        with pytest.raises(TypeError, match=msg):
            td + np.array([1])
        msg = '|'.join(["unsupported operand type\\(s\\) for \\+: 'numpy.ndarray' and 'Timedelta'", 'Concatenation operation is not implemented for NumPy arrays'])
        with pytest.raises(TypeError, match=msg):
            other + td
        msg = "unsupported operand type\\(s\\) for -: 'Timedelta' and 'int'"
        with pytest.raises(TypeError, match=msg):
            td - other
        msg = "unsupported operand type\\(s\\) for -: 'numpy.ndarray' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            other - td

    def test_td_rsub_nat(self):
        td = Timedelta(10, unit='d')
        result = NaT - td
        assert result is NaT
        result = np.datetime64('NaT') - td
        assert result is NaT

    def test_td_rsub_offset(self):
        result = offsets.Hour(1) - Timedelta(10, unit='d')
        assert isinstance(result, Timedelta)
        assert result == Timedelta(-239, unit='h')

    def test_td_sub_timedeltalike_object_dtype_array(self):
        arr = np.array([Timestamp('20130101 9:01'), Timestamp('20121230 9:02')])
        exp = np.array([Timestamp('20121231 9:01'), Timestamp('20121229 9:02')])
        res = arr - Timedelta('1D')
        tm.assert_numpy_array_equal(res, exp)

    def test_td_sub_mixed_most_timedeltalike_object_dtype_array(self):
        now = Timestamp('2021-11-09 09:54:00')
        arr = np.array([now, Timedelta('1D'), np.timedelta64(2, 'h')])
        exp = np.array([now - Timedelta('1D'), Timedelta('0D'), np.timedelta64(2, 'h') - Timedelta('1D')])
        res = arr - Timedelta('1D')
        tm.assert_numpy_array_equal(res, exp)

    def test_td_rsub_mixed_most_timedeltalike_object_dtype_array(self):
        now = Timestamp('2021-11-09 09:54:00')
        arr = np.array([now, Timedelta('1D'), np.timedelta64(2, 'h')])
        msg = "unsupported operand type\\(s\\) for \\-: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            Timedelta('1D') - arr

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_timedeltalike_object_dtype_array(self, op):
        arr = np.array([Timestamp('20130101 9:01'), Timestamp('20121230 9:02')])
        exp = np.array([Timestamp('20130102 9:01'), Timestamp('20121231 9:02')])
        res = op(arr, Timedelta('1D'))
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize('op', [operator.add, ops.radd])
    def test_td_add_mixed_timedeltalike_object_dtype_array(self, op):
        now = Timestamp('2021-11-09 09:54:00')
        arr = np.array([now, Timedelta('1D')])
        exp = np.array([now + Timedelta('1D'), Timedelta('2D')])
        res = op(arr, Timedelta('1D'))
        tm.assert_numpy_array_equal(res, exp)

    def test_td_add_sub_td64_ndarray(self):
        td = Timedelta('1 day')
        other = np.array([td.to_timedelta64()])
        expected = np.array([Timedelta('2 Days').to_timedelta64()])
        result = td + other
        tm.assert_numpy_array_equal(result, expected)
        result = other + td
        tm.assert_numpy_array_equal(result, expected)
        result = td - other
        tm.assert_numpy_array_equal(result, expected * 0)
        result = other - td
        tm.assert_numpy_array_equal(result, expected * 0)

    def test_td_add_sub_dt64_ndarray(self):
        td = Timedelta('1 day')
        other = np.array(['2000-01-01'], dtype='M8[ns]')
        expected = np.array(['2000-01-02'], dtype='M8[ns]')
        tm.assert_numpy_array_equal(td + other, expected)
        tm.assert_numpy_array_equal(other + td, expected)
        expected = np.array(['1999-12-31'], dtype='M8[ns]')
        tm.assert_numpy_array_equal(-td + other, expected)
        tm.assert_numpy_array_equal(other - td, expected)

    def test_td_add_sub_ndarray_0d(self):
        td = Timedelta('1 day')
        other = np.array(td.asm8)
        result = td + other
        assert isinstance(result, Timedelta)
        assert result == 2 * td
        result = other + td
        assert isinstance(result, Timedelta)
        assert result == 2 * td
        result = other - td
        assert isinstance(result, Timedelta)
        assert result == 0 * td
        result = td - other
        assert isinstance(result, Timedelta)
        assert result == 0 * td