import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestPeriodRange:

    @pytest.mark.parametrize('freq_offset, freq_period', [('D', 'D'), ('W', 'W'), ('QE', 'Q'), ('YE', 'Y')])
    def test_construction_from_string(self, freq_offset, freq_period):
        expected = date_range(start='2017-01-01', periods=5, freq=freq_offset, name='foo').to_period()
        start, end = (str(expected[0]), str(expected[-1]))
        result = period_range(start=start, end=end, freq=freq_period, name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(start=start, periods=5, freq=freq_period, name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(end=end, periods=5, freq=freq_period, name='foo')
        tm.assert_index_equal(result, expected)
        expected = PeriodIndex([], freq=freq_period, name='foo')
        result = period_range(start=start, periods=0, freq=freq_period, name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(end=end, periods=0, freq=freq_period, name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(start=end, end=start, freq=freq_period, name='foo')
        tm.assert_index_equal(result, expected)

    def test_construction_from_string_monthly(self):
        expected = date_range(start='2017-01-01', periods=5, freq='ME', name='foo').to_period()
        start, end = (str(expected[0]), str(expected[-1]))
        result = period_range(start=start, end=end, freq='M', name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(start=start, periods=5, freq='M', name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(end=end, periods=5, freq='M', name='foo')
        tm.assert_index_equal(result, expected)
        expected = PeriodIndex([], freq='M', name='foo')
        result = period_range(start=start, periods=0, freq='M', name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(end=end, periods=0, freq='M', name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(start=end, end=start, freq='M', name='foo')
        tm.assert_index_equal(result, expected)

    def test_construction_from_period(self):
        start, end = (Period('2017Q1', freq='Q'), Period('2018Q1', freq='Q'))
        expected = date_range(start='2017-03-31', end='2018-03-31', freq='ME', name='foo').to_period()
        result = period_range(start=start, end=end, freq='M', name='foo')
        tm.assert_index_equal(result, expected)
        start = Period('2017-1', freq='M')
        end = Period('2019-12', freq='M')
        expected = date_range(start='2017-01-31', end='2019-12-31', freq='QE', name='foo').to_period()
        result = period_range(start=start, end=end, freq='Q', name='foo')
        tm.assert_index_equal(result, expected)
        start = Period('2017Q1', freq='Q')
        end = Period('2018Q1', freq='Q')
        idx = period_range(start=start, end=end, freq='Q', name='foo')
        result = idx == idx.values
        expected = np.array([True, True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)
        expected = PeriodIndex([], freq='W', name='foo')
        result = period_range(start=start, periods=0, freq='W', name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(end=end, periods=0, freq='W', name='foo')
        tm.assert_index_equal(result, expected)
        result = period_range(start=end, end=start, freq='W', name='foo')
        tm.assert_index_equal(result, expected)

    def test_mismatched_start_end_freq_raises(self):
        depr_msg = 'Period with BDay freq is deprecated'
        msg = "'w' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            end_w = Period('2006-12-31', '1w')
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            start_b = Period('02-Apr-2005', 'B')
            end_b = Period('2005-05-01', 'B')
        msg = 'start and end must have same freq'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                period_range(start=start_b, end=end_w)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            period_range(start=start_b, end=end_b)