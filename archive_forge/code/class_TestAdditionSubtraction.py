from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
class TestAdditionSubtraction:

    @pytest.mark.parametrize('first, second, expected', [(Series([1, 2, 3], index=list('ABC'), name='x'), Series([2, 2, 2], index=list('ABD'), name='x'), Series([3.0, 4.0, np.nan, np.nan], index=list('ABCD'), name='x')), (Series([1, 2, 3], index=list('ABC'), name='x'), Series([2, 2, 2, 2], index=list('ABCD'), name='x'), Series([3, 4, 5, np.nan], index=list('ABCD'), name='x'))])
    def test_add_series(self, first, second, expected):
        tm.assert_series_equal(first + second, expected)
        tm.assert_series_equal(second + first, expected)

    @pytest.mark.parametrize('first, second, expected', [(pd.DataFrame({'x': [1, 2, 3]}, index=list('ABC')), pd.DataFrame({'x': [2, 2, 2]}, index=list('ABD')), pd.DataFrame({'x': [3.0, 4.0, np.nan, np.nan]}, index=list('ABCD'))), (pd.DataFrame({'x': [1, 2, 3]}, index=list('ABC')), pd.DataFrame({'x': [2, 2, 2, 2]}, index=list('ABCD')), pd.DataFrame({'x': [3, 4, 5, np.nan]}, index=list('ABCD')))])
    def test_add_frames(self, first, second, expected):
        tm.assert_frame_equal(first + second, expected)
        tm.assert_frame_equal(second + first, expected)

    def test_series_frame_radd_bug(self, fixed_now_ts):
        vals = Series([str(i) for i in range(5)])
        result = 'foo_' + vals
        expected = vals.map(lambda x: 'foo_' + x)
        tm.assert_series_equal(result, expected)
        frame = pd.DataFrame({'vals': vals})
        result = 'foo_' + frame
        expected = pd.DataFrame({'vals': vals.map(lambda x: 'foo_' + x)})
        tm.assert_frame_equal(result, expected)
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        fix_now = fixed_now_ts.to_pydatetime()
        msg = '|'.join(['unsupported operand type', 'Concatenation operation'])
        with pytest.raises(TypeError, match=msg):
            fix_now + ts
        with pytest.raises(TypeError, match=msg):
            ts + fix_now

    def test_datetime64_with_index(self):
        ser = Series(np.random.default_rng(2).standard_normal(5))
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)
        ser = Series(date_range('20130101', periods=5), index=date_range('20130101', periods=5))
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)
        msg = 'cannot subtract PeriodArray from DatetimeArray'
        with pytest.raises(TypeError, match=msg):
            result = ser - ser.index.to_period()
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=date_range('20130101', periods=5))
        df['date'] = pd.Timestamp('20130102')
        df['expected'] = df['date'] - df.index.to_series()
        df['result'] = df['date'] - df.index
        tm.assert_series_equal(df['result'], df['expected'], check_names=False)

    def test_frame_operators(self, float_frame):
        frame = float_frame
        garbage = np.random.default_rng(2).random(4)
        colSeries = Series(garbage, index=np.array(frame.columns))
        idSum = frame + frame
        seriesSum = frame + colSeries
        for col, series in idSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] * 2
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)
        for col, series in seriesSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] + colSeries[col]
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

    def test_frame_operators_col_align(self, float_frame):
        frame2 = pd.DataFrame(float_frame, columns=['D', 'C', 'B', 'A'])
        added = frame2 + frame2
        expected = frame2 * 2
        tm.assert_frame_equal(added, expected)

    def test_frame_operators_none_to_nan(self):
        df = pd.DataFrame({'a': ['a', None, 'b']})
        tm.assert_frame_equal(df + df, pd.DataFrame({'a': ['aa', np.nan, 'bb']}))

    @pytest.mark.parametrize('dtype', ('float', 'int64'))
    def test_frame_operators_empty_like(self, dtype):
        frames = [pd.DataFrame(dtype=dtype), pd.DataFrame(columns=['A'], dtype=dtype), pd.DataFrame(index=[0], dtype=dtype)]
        for df in frames:
            assert (df + df).equals(df)
            tm.assert_frame_equal(df + df, df)

    @pytest.mark.parametrize('func', [lambda x: x * 2, lambda x: x[::2], lambda x: 5], ids=['multiply', 'slice', 'constant'])
    def test_series_operators_arithmetic(self, all_arithmetic_functions, func):
        op = all_arithmetic_functions
        series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        other = func(series)
        compare_op(series, other, op)

    @pytest.mark.parametrize('func', [lambda x: x + 1, lambda x: 5], ids=['add', 'constant'])
    def test_series_operators_compare(self, comparison_op, func):
        op = comparison_op
        series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        other = func(series)
        compare_op(series, other, op)

    @pytest.mark.parametrize('func', [lambda x: x * 2, lambda x: x[::2], lambda x: 5], ids=['multiply', 'slice', 'constant'])
    def test_divmod(self, func):
        series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        other = func(series)
        results = divmod(series, other)
        if isinstance(other, abc.Iterable) and len(series) != len(other):
            other_np = []
            for n in other:
                other_np.append(n)
                other_np.append(np.nan)
        else:
            other_np = other
        other_np = np.asarray(other_np)
        with np.errstate(all='ignore'):
            expecteds = divmod(series.values, np.asarray(other_np))
        for result, expected in zip(results, expecteds):
            tm.assert_almost_equal(np.asarray(result), expected)
            assert result.name == series.name
            tm.assert_index_equal(result.index, series.index._with_freq(None))

    def test_series_divmod_zero(self):
        tser = Series(np.arange(1, 11, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        other = tser * 0
        result = divmod(tser, other)
        exp1 = Series([np.inf] * len(tser), index=tser.index, name='ts')
        exp2 = Series([np.nan] * len(tser), index=tser.index, name='ts')
        tm.assert_series_equal(result[0], exp1)
        tm.assert_series_equal(result[1], exp2)