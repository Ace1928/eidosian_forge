import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestCategoricalOps:

    @pytest.mark.parametrize('categories', [['a', 'b'], [0, 1], [Timestamp('2019'), Timestamp('2020')]])
    def test_not_equal_with_na(self, categories):
        c1 = Categorical.from_codes([-1, 0], categories=categories)
        c2 = Categorical.from_codes([0, 1], categories=categories)
        result = c1 != c2
        assert result.all()

    def test_compare_frame(self):
        data = ['a', 'b', 2, 'a']
        cat = Categorical(data)
        df = DataFrame(cat)
        result = cat == df.T
        expected = DataFrame([[True, True, True, True]])
        tm.assert_frame_equal(result, expected)
        result = cat[::-1] != df.T
        expected = DataFrame([[False, True, True, False]])
        tm.assert_frame_equal(result, expected)

    def test_compare_frame_raises(self, comparison_op):
        op = comparison_op
        cat = Categorical(['a', 'b', 2, 'a'])
        df = DataFrame(cat)
        msg = 'Unable to coerce to Series, length must be 1: given 4'
        with pytest.raises(ValueError, match=msg):
            op(cat, df)

    def test_datetime_categorical_comparison(self):
        dt_cat = Categorical(date_range('2014-01-01', periods=3), ordered=True)
        tm.assert_numpy_array_equal(dt_cat > dt_cat[0], np.array([False, True, True]))
        tm.assert_numpy_array_equal(dt_cat[0] < dt_cat, np.array([False, True, True]))

    def test_reflected_comparison_with_scalars(self):
        cat = Categorical([1, 2, 3], ordered=True)
        tm.assert_numpy_array_equal(cat > cat[0], np.array([False, True, True]))
        tm.assert_numpy_array_equal(cat[0] < cat, np.array([False, True, True]))

    def test_comparison_with_unknown_scalars(self):
        cat = Categorical([1, 2, 3], ordered=True)
        msg = 'Invalid comparison between dtype=category and int'
        with pytest.raises(TypeError, match=msg):
            cat < 4
        with pytest.raises(TypeError, match=msg):
            cat > 4
        with pytest.raises(TypeError, match=msg):
            4 < cat
        with pytest.raises(TypeError, match=msg):
            4 > cat
        tm.assert_numpy_array_equal(cat == 4, np.array([False, False, False]))
        tm.assert_numpy_array_equal(cat != 4, np.array([True, True, True]))

    def test_comparison_with_tuple(self):
        cat = Categorical(np.array(['foo', (0, 1), 3, (0, 1)], dtype=object))
        result = cat == 'foo'
        expected = np.array([True, False, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)
        result = cat == (0, 1)
        expected = np.array([False, True, False, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)
        result = cat != (0, 1)
        tm.assert_numpy_array_equal(result, ~expected)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_comparison_of_ordered_categorical_with_nan_to_scalar(self, compare_operators_no_eq_ne):
        cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
        scalar = 2
        expected = getattr(np.array(cat), compare_operators_no_eq_ne)(scalar)
        actual = getattr(cat, compare_operators_no_eq_ne)(scalar)
        tm.assert_numpy_array_equal(actual, expected)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_comparison_of_ordered_categorical_with_nan_to_listlike(self, compare_operators_no_eq_ne):
        cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
        other = Categorical([2, 2, 2, 2], categories=[1, 2, 3], ordered=True)
        expected = getattr(np.array(cat), compare_operators_no_eq_ne)(2)
        actual = getattr(cat, compare_operators_no_eq_ne)(other)
        tm.assert_numpy_array_equal(actual, expected)

    @pytest.mark.parametrize('data,reverse,base', [(list('abc'), list('cba'), list('bbb')), ([1, 2, 3], [3, 2, 1], [2, 2, 2])])
    def test_comparisons(self, data, reverse, base):
        cat_rev = Series(Categorical(data, categories=reverse, ordered=True))
        cat_rev_base = Series(Categorical(base, categories=reverse, ordered=True))
        cat = Series(Categorical(data, ordered=True))
        cat_base = Series(Categorical(base, categories=cat.cat.categories, ordered=True))
        s = Series(base, dtype=object if base == list('bbb') else None)
        a = np.array(base)
        res_rev = cat_rev > cat_rev_base
        exp_rev = Series([True, False, False])
        tm.assert_series_equal(res_rev, exp_rev)
        res_rev = cat_rev < cat_rev_base
        exp_rev = Series([False, False, True])
        tm.assert_series_equal(res_rev, exp_rev)
        res = cat > cat_base
        exp = Series([False, False, True])
        tm.assert_series_equal(res, exp)
        scalar = base[1]
        res = cat > scalar
        exp = Series([False, False, True])
        exp2 = cat.values > scalar
        tm.assert_series_equal(res, exp)
        tm.assert_numpy_array_equal(res.values, exp2)
        res_rev = cat_rev > scalar
        exp_rev = Series([True, False, False])
        exp_rev2 = cat_rev.values > scalar
        tm.assert_series_equal(res_rev, exp_rev)
        tm.assert_numpy_array_equal(res_rev.values, exp_rev2)
        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):
            cat > cat_rev
        msg = "Cannot compare a Categorical for op __gt__ with type <class 'numpy\\.ndarray'>"
        with pytest.raises(TypeError, match=msg):
            cat > s
        with pytest.raises(TypeError, match=msg):
            cat_rev > s
        with pytest.raises(TypeError, match=msg):
            cat > a
        with pytest.raises(TypeError, match=msg):
            cat_rev > a
        with pytest.raises(TypeError, match=msg):
            s < cat
        with pytest.raises(TypeError, match=msg):
            s < cat_rev
        with pytest.raises(TypeError, match=msg):
            a < cat
        with pytest.raises(TypeError, match=msg):
            a < cat_rev

    @pytest.mark.parametrize('ctor', [lambda *args, **kwargs: Categorical(*args, **kwargs), lambda *args, **kwargs: Series(Categorical(*args, **kwargs))])
    def test_unordered_different_order_equal(self, ctor):
        c1 = ctor(['a', 'b'], categories=['a', 'b'], ordered=False)
        c2 = ctor(['a', 'b'], categories=['b', 'a'], ordered=False)
        assert (c1 == c2).all()
        c1 = ctor(['a', 'b'], categories=['a', 'b'], ordered=False)
        c2 = ctor(['b', 'a'], categories=['b', 'a'], ordered=False)
        assert (c1 != c2).all()
        c1 = ctor(['a', 'a'], categories=['a', 'b'], ordered=False)
        c2 = ctor(['b', 'b'], categories=['b', 'a'], ordered=False)
        assert (c1 != c2).all()
        c1 = ctor(['a', 'a'], categories=['a', 'b'], ordered=False)
        c2 = ctor(['a', 'b'], categories=['b', 'a'], ordered=False)
        result = c1 == c2
        tm.assert_numpy_array_equal(np.array(result), np.array([True, False]))

    def test_unordered_different_categories_raises(self):
        c1 = Categorical(['a', 'b'], categories=['a', 'b'], ordered=False)
        c2 = Categorical(['a', 'c'], categories=['c', 'a'], ordered=False)
        with pytest.raises(TypeError, match='Categoricals can only be compared'):
            c1 == c2

    def test_compare_different_lengths(self):
        c1 = Categorical([], categories=['a', 'b'])
        c2 = Categorical([], categories=['a'])
        msg = "Categoricals can only be compared if 'categories' are the same."
        with pytest.raises(TypeError, match=msg):
            c1 == c2

    def test_compare_unordered_different_order(self):
        a = Categorical(['a'], categories=['a', 'b'])
        b = Categorical(['b'], categories=['b', 'a'])
        assert not a.equals(b)

    def test_numeric_like_ops(self):
        df = DataFrame({'value': np.random.default_rng(2).integers(0, 10000, 100)})
        labels = [f'{i} - {i + 499}' for i in range(0, 10000, 500)]
        cat_labels = Categorical(labels, labels)
        df = df.sort_values(by=['value'], ascending=True)
        df['value_group'] = pd.cut(df.value, range(0, 10500, 500), right=False, labels=cat_labels)
        for op, str_rep in [('__add__', '\\+'), ('__sub__', '-'), ('__mul__', '\\*'), ('__truediv__', '/')]:
            msg = f'Series cannot perform the operation {str_rep}|unsupported operand'
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)(df)
        s = df['value_group']
        for op in ['kurt', 'skew', 'var', 'std', 'mean', 'sum', 'median']:
            msg = f"does not support reduction '{op}'"
            with pytest.raises(TypeError, match=msg):
                getattr(s, op)(numeric_only=False)

    def test_numeric_like_ops_series(self):
        s = Series(Categorical([1, 2, 3, 4]))
        with pytest.raises(TypeError, match="does not support reduction 'sum'"):
            np.sum(s)

    @pytest.mark.parametrize('op, str_rep', [('__add__', '\\+'), ('__sub__', '-'), ('__mul__', '\\*'), ('__truediv__', '/')])
    def test_numeric_like_ops_series_arith(self, op, str_rep):
        s = Series(Categorical([1, 2, 3, 4]))
        msg = f'Series cannot perform the operation {str_rep}|unsupported operand'
        with pytest.raises(TypeError, match=msg):
            getattr(s, op)(2)

    def test_numeric_like_ops_series_invalid(self):
        s = Series(Categorical([1, 2, 3, 4]))
        msg = 'Object with dtype category cannot perform the numpy op log'
        with pytest.raises(TypeError, match=msg):
            np.log(s)