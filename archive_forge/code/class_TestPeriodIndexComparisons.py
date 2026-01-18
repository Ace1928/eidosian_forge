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
class TestPeriodIndexComparisons:

    def test_pi_cmp_period(self):
        idx = period_range('2007-01', periods=20, freq='M')
        per = idx[10]
        result = idx < per
        exp = idx.values < idx.values[10]
        tm.assert_numpy_array_equal(result, exp)
        result = idx.values.reshape(10, 2) < per
        tm.assert_numpy_array_equal(result, exp.reshape(10, 2))
        result = idx < np.array(per)
        tm.assert_numpy_array_equal(result, exp)

    def test_parr_cmp_period_scalar2(self, box_with_array):
        pi = period_range('2000-01-01', periods=10, freq='D')
        val = pi[3]
        expected = [x > val for x in pi]
        ser = tm.box_expected(pi, box_with_array)
        xbox = get_upcast_box(ser, val, True)
        expected = tm.box_expected(expected, xbox)
        result = ser > val
        tm.assert_equal(result, expected)
        val = pi[5]
        result = ser > val
        expected = [x > val for x in pi]
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('freq', ['M', '2M', '3M'])
    def test_parr_cmp_period_scalar(self, freq, box_with_array):
        base = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq=freq)
        base = tm.box_expected(base, box_with_array)
        per = Period('2011-02', freq=freq)
        xbox = get_upcast_box(base, per, True)
        exp = np.array([False, True, False, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base == per, exp)
        tm.assert_equal(per == base, exp)
        exp = np.array([True, False, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base != per, exp)
        tm.assert_equal(per != base, exp)
        exp = np.array([False, False, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base > per, exp)
        tm.assert_equal(per < base, exp)
        exp = np.array([True, False, False, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base < per, exp)
        tm.assert_equal(per > base, exp)
        exp = np.array([False, True, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base >= per, exp)
        tm.assert_equal(per <= base, exp)
        exp = np.array([True, True, False, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base <= per, exp)
        tm.assert_equal(per >= base, exp)

    @pytest.mark.parametrize('freq', ['M', '2M', '3M'])
    def test_parr_cmp_pi(self, freq, box_with_array):
        base = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq=freq)
        base = tm.box_expected(base, box_with_array)
        idx = PeriodIndex(['2011-02', '2011-01', '2011-03', '2011-05'], freq=freq)
        xbox = get_upcast_box(base, idx, True)
        exp = np.array([False, False, True, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base == idx, exp)
        exp = np.array([True, True, False, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base != idx, exp)
        exp = np.array([False, True, False, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base > idx, exp)
        exp = np.array([True, False, False, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base < idx, exp)
        exp = np.array([False, True, True, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base >= idx, exp)
        exp = np.array([True, False, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base <= idx, exp)

    @pytest.mark.parametrize('freq', ['M', '2M', '3M'])
    def test_parr_cmp_pi_mismatched_freq(self, freq, box_with_array):
        base = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq=freq)
        base = tm.box_expected(base, box_with_array)
        msg = f'Invalid comparison between dtype=period\\[{freq}\\] and Period'
        with pytest.raises(TypeError, match=msg):
            base <= Period('2011', freq='Y')
        with pytest.raises(TypeError, match=msg):
            Period('2011', freq='Y') >= base
        idx = PeriodIndex(['2011', '2012', '2013', '2014'], freq='Y')
        rev_msg = 'Invalid comparison between dtype=period\\[Y-DEC\\] and PeriodArray'
        idx_msg = rev_msg if box_with_array in [tm.to_array, pd.array] else msg
        with pytest.raises(TypeError, match=idx_msg):
            base <= idx
        msg = f'Invalid comparison between dtype=period\\[{freq}\\] and Period'
        with pytest.raises(TypeError, match=msg):
            base <= Period('2011', freq='4M')
        with pytest.raises(TypeError, match=msg):
            Period('2011', freq='4M') >= base
        idx = PeriodIndex(['2011', '2012', '2013', '2014'], freq='4M')
        rev_msg = 'Invalid comparison between dtype=period\\[4M\\] and PeriodArray'
        idx_msg = rev_msg if box_with_array in [tm.to_array, pd.array] else msg
        with pytest.raises(TypeError, match=idx_msg):
            base <= idx

    @pytest.mark.parametrize('freq', ['M', '2M', '3M'])
    def test_pi_cmp_nat(self, freq):
        idx1 = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-05'], freq=freq)
        per = idx1[1]
        result = idx1 > per
        exp = np.array([False, False, False, True])
        tm.assert_numpy_array_equal(result, exp)
        result = per < idx1
        tm.assert_numpy_array_equal(result, exp)
        result = idx1 == pd.NaT
        exp = np.array([False, False, False, False])
        tm.assert_numpy_array_equal(result, exp)
        result = pd.NaT == idx1
        tm.assert_numpy_array_equal(result, exp)
        result = idx1 != pd.NaT
        exp = np.array([True, True, True, True])
        tm.assert_numpy_array_equal(result, exp)
        result = pd.NaT != idx1
        tm.assert_numpy_array_equal(result, exp)
        idx2 = PeriodIndex(['2011-02', '2011-01', '2011-04', 'NaT'], freq=freq)
        result = idx1 < idx2
        exp = np.array([True, False, False, False])
        tm.assert_numpy_array_equal(result, exp)
        result = idx1 == idx2
        exp = np.array([False, False, False, False])
        tm.assert_numpy_array_equal(result, exp)
        result = idx1 != idx2
        exp = np.array([True, True, True, True])
        tm.assert_numpy_array_equal(result, exp)
        result = idx1 == idx1
        exp = np.array([True, True, False, True])
        tm.assert_numpy_array_equal(result, exp)
        result = idx1 != idx1
        exp = np.array([False, False, True, False])
        tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize('freq', ['M', '2M', '3M'])
    def test_pi_cmp_nat_mismatched_freq_raises(self, freq):
        idx1 = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-05'], freq=freq)
        diff = PeriodIndex(['2011-02', '2011-01', '2011-04', 'NaT'], freq='4M')
        msg = f'Invalid comparison between dtype=period\\[{freq}\\] and PeriodArray'
        with pytest.raises(TypeError, match=msg):
            idx1 > diff
        result = idx1 == diff
        expected = np.array([False, False, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', [object, None])
    def test_comp_nat(self, dtype):
        left = PeriodIndex([Period('2011-01-01'), pd.NaT, Period('2011-01-03')])
        right = PeriodIndex([pd.NaT, pd.NaT, Period('2011-01-03')])
        if dtype is not None:
            left = left.astype(dtype)
            right = right.astype(dtype)
        result = left == right
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = left != right
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(left == pd.NaT, expected)
        tm.assert_numpy_array_equal(pd.NaT == right, expected)
        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(left != pd.NaT, expected)
        tm.assert_numpy_array_equal(pd.NaT != left, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(left < pd.NaT, expected)
        tm.assert_numpy_array_equal(pd.NaT > left, expected)