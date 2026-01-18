import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
class TestIntervalIndexRendering:

    @pytest.mark.parametrize('constructor,expected', [(Series, '(0.0, 1.0]    a\nNaN           b\n(2.0, 3.0]    c\ndtype: object'), (DataFrame, '            0\n(0.0, 1.0]  a\nNaN         b\n(2.0, 3.0]  c')])
    def test_repr_missing(self, constructor, expected, using_infer_string, request):
        if using_infer_string and constructor is Series:
            request.applymarker(pytest.mark.xfail(reason='repr different'))
        index = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        obj = constructor(list('abc'), index=index)
        result = repr(obj)
        assert result == expected

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason='repr different')
    def test_repr_floats(self):
        markers = Series(['foo', 'bar'], index=IntervalIndex([Interval(left, right) for left, right in zip(Index([329.973, 345.137], dtype='float64'), Index([345.137, 360.191], dtype='float64'))]))
        result = str(markers)
        expected = '(329.973, 345.137]    foo\n(345.137, 360.191]    bar\ndtype: object'
        assert result == expected

    @pytest.mark.parametrize('tuples, closed, expected_data', [([(0, 1), (1, 2), (2, 3)], 'left', ['[0, 1)', '[1, 2)', '[2, 3)']), ([(0.5, 1.0), np.nan, (2.0, 3.0)], 'right', ['(0.5, 1.0]', 'NaN', '(2.0, 3.0]']), ([(Timestamp('20180101'), Timestamp('20180102')), np.nan, (Timestamp('20180102'), Timestamp('20180103'))], 'both', ['[2018-01-01 00:00:00, 2018-01-02 00:00:00]', 'NaN', '[2018-01-02 00:00:00, 2018-01-03 00:00:00]']), ([(Timedelta('0 days'), Timedelta('1 days')), (Timedelta('1 days'), Timedelta('2 days')), np.nan], 'neither', ['(0 days 00:00:00, 1 days 00:00:00)', '(1 days 00:00:00, 2 days 00:00:00)', 'NaN'])])
    def test_get_values_for_csv(self, tuples, closed, expected_data):
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index._get_values_for_csv(na_rep='NaN')
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)

    def test_timestamp_with_timezone(self, unit):
        left = DatetimeIndex(['2020-01-01'], dtype=f'M8[{unit}, UTC]')
        right = DatetimeIndex(['2020-01-02'], dtype=f'M8[{unit}, UTC]')
        index = IntervalIndex.from_arrays(left, right)
        result = repr(index)
        expected = f"IntervalIndex([(2020-01-01 00:00:00+00:00, 2020-01-02 00:00:00+00:00]], dtype='interval[datetime64[{unit}, UTC], right]')"
        assert result == expected