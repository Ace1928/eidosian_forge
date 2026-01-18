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
class TestDatetimeIndexComparisons:

    def test_comparators(self, comparison_op):
        index = date_range('2020-01-01', periods=10)
        element = index[len(index) // 2]
        element = Timestamp(element).to_datetime64()
        arr = np.array(index)
        arr_result = comparison_op(arr, element)
        index_result = comparison_op(index, element)
        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    def test_dti_cmp_datetimelike(self, other, tz_naive_fixture):
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                pytest.skip(f'{type(other).__name__} is not tz aware')
            other = localize_pydatetime(other, dti.tzinfo)
        result = dti == other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dti > other
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = dti >= other
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = dti < other
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dti <= other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', [None, object])
    def test_dti_cmp_nat(self, dtype, box_with_array):
        left = DatetimeIndex([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')])
        right = DatetimeIndex([NaT, NaT, Timestamp('2011-01-03')])
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        xbox = get_upcast_box(left, right, True)
        lhs, rhs = (left, right)
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result = rhs == lhs
        expected = np.array([False, False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        result = lhs != rhs
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs == NaT, expected)
        tm.assert_equal(NaT == rhs, expected)
        expected = np.array([True, True, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs != NaT, expected)
        tm.assert_equal(NaT != lhs, expected)
        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs < NaT, expected)
        tm.assert_equal(NaT > lhs, expected)

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self):
        fidx1 = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        fidx2 = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])
        didx1 = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        didx2 = DatetimeIndex(['2014-02-01', '2014-03-01', NaT, NaT, '2014-06-01', '2014-07-01'])
        darr = np.array([np.datetime64('2014-02-01 00:00'), np.datetime64('2014-03-01 00:00'), np.datetime64('nat'), np.datetime64('nat'), np.datetime64('2014-06-01 00:00'), np.datetime64('2014-07-01 00:00')])
        cases = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]
        with tm.assert_produces_warning(None):
            for idx1, idx2 in cases:
                result = idx1 < idx2
                expected = np.array([True, False, False, False, True, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx2 > idx1
                expected = np.array([True, False, False, False, True, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 <= idx2
                expected = np.array([True, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx2 >= idx1
                expected = np.array([True, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 == idx2
                expected = np.array([False, False, False, False, False, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 != idx2
                expected = np.array([True, True, True, True, True, False])
                tm.assert_numpy_array_equal(result, expected)
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, np.nan), (didx1, NaT)]:
                result = idx1 < val
                expected = np.array([False, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 <= val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 == val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 != val
                expected = np.array([True, True, True, True, True, True])
                tm.assert_numpy_array_equal(result, expected)
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, 3), (didx1, datetime(2014, 3, 1))]:
                result = idx1 < val
                expected = np.array([True, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                expected = np.array([False, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 <= val
                expected = np.array([True, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                expected = np.array([False, False, True, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 == val
                expected = np.array([False, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 != val
                expected = np.array([True, True, False, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

    def test_comparison_tzawareness_compat(self, comparison_op, box_with_array):
        op = comparison_op
        box = box_with_array
        dr = date_range('2016-01-01', periods=6)
        dz = dr.tz_localize('US/Pacific')
        dr = tm.box_expected(dr, box)
        dz = tm.box_expected(dz, box)
        if box is pd.DataFrame:
            tolist = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist = list
        if op not in [operator.eq, operator.ne]:
            msg = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and (Timestamp|DatetimeArray|list|ndarray)'
            with pytest.raises(TypeError, match=msg):
                op(dr, dz)
            with pytest.raises(TypeError, match=msg):
                op(dr, tolist(dz))
            with pytest.raises(TypeError, match=msg):
                op(dr, np.array(tolist(dz), dtype=object))
            with pytest.raises(TypeError, match=msg):
                op(dz, dr)
            with pytest.raises(TypeError, match=msg):
                op(dz, tolist(dr))
            with pytest.raises(TypeError, match=msg):
                op(dz, np.array(tolist(dr), dtype=object))
        assert np.all(dr == dr)
        assert np.all(dr == tolist(dr))
        assert np.all(tolist(dr) == dr)
        assert np.all(np.array(tolist(dr), dtype=object) == dr)
        assert np.all(dr == np.array(tolist(dr), dtype=object))
        assert np.all(dz == dz)
        assert np.all(dz == tolist(dz))
        assert np.all(tolist(dz) == dz)
        assert np.all(np.array(tolist(dz), dtype=object) == dz)
        assert np.all(dz == np.array(tolist(dz), dtype=object))

    def test_comparison_tzawareness_compat_scalars(self, comparison_op, box_with_array):
        op = comparison_op
        dr = date_range('2016-01-01', periods=6)
        dz = dr.tz_localize('US/Pacific')
        dr = tm.box_expected(dr, box_with_array)
        dz = tm.box_expected(dz, box_with_array)
        ts = Timestamp('2000-03-14 01:59')
        ts_tz = Timestamp('2000-03-14 01:59', tz='Europe/Amsterdam')
        assert np.all(dr > ts)
        msg = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and Timestamp'
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dr, ts_tz)
        assert np.all(dz > ts_tz)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dz, ts)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(ts, dz)

    @pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    @pytest.mark.filterwarnings('ignore:elementwise comp:DeprecationWarning')
    def test_scalar_comparison_tzawareness(self, comparison_op, other, tz_aware_fixture, box_with_array):
        op = comparison_op
        tz = tz_aware_fixture
        dti = date_range('2016-01-01', periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        xbox = get_upcast_box(dtarr, other, True)
        if op in [operator.eq, operator.ne]:
            exbool = op is operator.ne
            expected = np.array([exbool, exbool], dtype=bool)
            expected = tm.box_expected(expected, xbox)
            result = op(dtarr, other)
            tm.assert_equal(result, expected)
            result = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            msg = f'Invalid comparison between dtype=datetime64\\[ns, .*\\] and {type(other).__name__}'
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    def test_nat_comparison_tzawareness(self, comparison_op):
        op = comparison_op
        dti = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        expected = np.array([op == operator.ne] * len(dti))
        result = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)
        result = op(dti.tz_localize('US/Pacific'), NaT)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_str(self, tz_naive_fixture):
        tz = tz_naive_fixture
        rng = date_range('1/1/2000', periods=10, tz=tz)
        other = '1/1/2000'
        result = rng == other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng != other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng < other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result = rng <= other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng > other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng >= other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_list(self):
        rng = date_range('1/1/2000', periods=10)
        result = rng == list(rng)
        expected = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other', [pd.timedelta_range('1D', periods=10), pd.timedelta_range('1D', periods=10).to_series(), pd.timedelta_range('1D', periods=10).asi8.view('m8[ns]')], ids=lambda x: type(x).__name__)
    def test_dti_cmp_tdi_tzawareness(self, other):
        dti = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        result = dti == other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result = dti != other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        msg = 'Invalid comparison between'
        with pytest.raises(TypeError, match=msg):
            dti < other
        with pytest.raises(TypeError, match=msg):
            dti <= other
        with pytest.raises(TypeError, match=msg):
            dti > other
        with pytest.raises(TypeError, match=msg):
            dti >= other

    def test_dti_cmp_object_dtype(self):
        dti = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        other = dti.astype('O')
        result = dti == other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        other = dti.tz_localize(None)
        result = dti != other
        tm.assert_numpy_array_equal(result, expected)
        other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        result = dti == other
        expected = np.array([True] * 5 + [False] * 5)
        tm.assert_numpy_array_equal(result, expected)
        msg = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            dti >= other