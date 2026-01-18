from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestSeriesComparison:

    def test_comparison_different_length(self):
        a = Series(['a', 'b', 'c'])
        b = Series(['b', 'a'])
        msg = 'only compare identically-labeled Series'
        with pytest.raises(ValueError, match=msg):
            a < b
        a = Series([1, 2])
        b = Series([2, 3, 4])
        with pytest.raises(ValueError, match=msg):
            a == b

    @pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_ser_flex_cmp_return_dtypes(self, opname):
        ser = Series([1, 3, 2], index=range(3))
        const = 2
        result = getattr(ser, opname)(const).dtypes
        expected = np.dtype('bool')
        assert result == expected

    @pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_ser_flex_cmp_return_dtypes_empty(self, opname):
        ser = Series([1, 3, 2], index=range(3))
        empty = ser.iloc[:0]
        const = 2
        result = getattr(empty, opname)(const).dtypes
        expected = np.dtype('bool')
        assert result == expected

    @pytest.mark.parametrize('names', [(None, None, None), ('foo', 'bar', None), ('baz', 'baz', 'baz')])
    def test_ser_cmp_result_names(self, names, comparison_op):
        op = comparison_op
        dti = date_range('1949-06-07 03:00:00', freq='h', periods=5, name=names[0])
        ser = Series(dti).rename(names[1])
        result = op(ser, dti)
        assert result.name == names[2]
        dti = dti.tz_localize('US/Central')
        dti = pd.DatetimeIndex(dti, freq='infer')
        ser = Series(dti).rename(names[1])
        result = op(ser, dti)
        assert result.name == names[2]
        tdi = dti - dti.shift(1)
        ser = Series(tdi).rename(names[1])
        result = op(ser, tdi)
        assert result.name == names[2]
        if op in [operator.eq, operator.ne]:
            ii = pd.interval_range(start=0, periods=5, name=names[0])
            ser = Series(ii).rename(names[1])
            result = op(ser, ii)
            assert result.name == names[2]
        if op in [operator.eq, operator.ne]:
            cidx = tdi.astype('category')
            ser = Series(cidx).rename(names[1])
            result = op(ser, cidx)
            assert result.name == names[2]

    def test_comparisons(self, using_infer_string):
        s = Series(['a', 'b', 'c'])
        s2 = Series([False, True, False])
        exp = Series([False, False, False])
        if using_infer_string:
            import pyarrow as pa
            msg = 'has no kernel'
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                s == s2
            with tm.assert_produces_warning(DeprecationWarning, match='comparison', check_stacklevel=False):
                with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                    s2 == s
        else:
            tm.assert_series_equal(s == s2, exp)
            tm.assert_series_equal(s2 == s, exp)

    def test_categorical_comparisons(self):
        a = Series(list('abc'), dtype='category')
        b = Series(list('abc'), dtype='object')
        c = Series(['a', 'b', 'cc'], dtype='object')
        d = Series(list('acb'), dtype='object')
        e = Categorical(list('abc'))
        f = Categorical(list('acb'))
        assert not (a == 'a').all()
        assert ((a != 'a') == ~(a == 'a')).all()
        assert not ('a' == a).all()
        assert (a == 'a')[0]
        assert ('a' == a)[0]
        assert not ('a' != a)[0]
        assert (a == a).all()
        assert not (a != a).all()
        assert (a == list(a)).all()
        assert (a == b).all()
        assert (b == a).all()
        assert (~(a == b) == (a != b)).all()
        assert (~(b == a) == (b != a)).all()
        assert not (a == c).all()
        assert not (c == a).all()
        assert not (a == d).all()
        assert not (d == a).all()
        assert (a == e).all()
        assert (e == a).all()
        assert not (a == f).all()
        assert not (f == a).all()
        assert (~(a == e) == (a != e)).all()
        assert (~(e == a) == (e != a)).all()
        assert (~(a == f) == (a != f)).all()
        assert (~(f == a) == (f != a)).all()
        msg = 'can only compare equality or not'
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            b > a

    def test_unequal_categorical_comparison_raises_type_error(self):
        cat = Series(Categorical(list('abc')))
        msg = 'can only compare equality or not'
        with pytest.raises(TypeError, match=msg):
            cat > 'b'
        cat = Series(Categorical(list('abc'), ordered=False))
        with pytest.raises(TypeError, match=msg):
            cat > 'b'
        cat = Series(Categorical(list('abc'), ordered=True))
        msg = 'Invalid comparison between dtype=category and str'
        with pytest.raises(TypeError, match=msg):
            cat < 'd'
        with pytest.raises(TypeError, match=msg):
            cat > 'd'
        with pytest.raises(TypeError, match=msg):
            'd' < cat
        with pytest.raises(TypeError, match=msg):
            'd' > cat
        tm.assert_series_equal(cat == 'd', Series([False, False, False]))
        tm.assert_series_equal(cat != 'd', Series([True, True, True]))

    def test_comparison_tuples(self):
        s = Series([(1, 1), (1, 2)])
        result = s == (1, 2)
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)
        result = s != (1, 2)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)
        result = s == (0, 0)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)
        result = s != (0, 0)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)
        s = Series([(1, 1), (1, 1)])
        result = s == (1, 1)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)
        result = s != (1, 1)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_frozenset(self):
        ser = Series([frozenset([1]), frozenset([1, 2])])
        result = ser == frozenset([1])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_operators_with_nas(self, comparison_op):
        ser = Series(bdate_range('1/1/2000', periods=10), dtype=object)
        ser[::2] = np.nan
        val = ser[5]
        result = comparison_op(ser, val)
        expected = comparison_op(ser.dropna(), val).reindex(ser.index)
        msg = 'Downcasting object dtype arrays'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            if comparison_op is operator.ne:
                expected = expected.fillna(True).astype(bool)
            else:
                expected = expected.fillna(False).astype(bool)
        tm.assert_series_equal(result, expected)

    def test_ne(self):
        ts = Series([3, 4, 5, 6, 7], [3, 4, 5, 6, 7], dtype=float)
        expected = np.array([True, True, False, True, True])
        tm.assert_numpy_array_equal(ts.index != 5, expected)
        tm.assert_numpy_array_equal(~(ts.index == 5), expected)

    @pytest.mark.parametrize('left, right', [(Series([1, 2, 3], index=list('ABC'), name='x'), Series([2, 2, 2], index=list('ABD'), name='x')), (Series([1, 2, 3], index=list('ABC'), name='x'), Series([2, 2, 2, 2], index=list('ABCD'), name='x'))])
    def test_comp_ops_df_compat(self, left, right, frame_or_series):
        if frame_or_series is not Series:
            msg = f'Can only compare identically-labeled \\(both index and columns\\) {frame_or_series.__name__} objects'
            left = left.to_frame()
            right = right.to_frame()
        else:
            msg = f'Can only compare identically-labeled {frame_or_series.__name__} objects'
        with pytest.raises(ValueError, match=msg):
            left == right
        with pytest.raises(ValueError, match=msg):
            right == left
        with pytest.raises(ValueError, match=msg):
            left != right
        with pytest.raises(ValueError, match=msg):
            right != left
        with pytest.raises(ValueError, match=msg):
            left < right
        with pytest.raises(ValueError, match=msg):
            right < left

    def test_compare_series_interval_keyword(self):
        ser = Series(['IntervalA', 'IntervalB', 'IntervalC'])
        result = ser == 'IntervalA'
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)