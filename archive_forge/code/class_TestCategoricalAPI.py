import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
class TestCategoricalAPI:

    def test_to_list_deprecated(self):
        cat1 = Categorical(list('acb'), ordered=False)
        msg = 'Categorical.to_list is deprecated and will be removed'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            cat1.to_list()

    def test_ordered_api(self):
        cat1 = Categorical(list('acb'), ordered=False)
        tm.assert_index_equal(cat1.categories, Index(['a', 'b', 'c']))
        assert not cat1.ordered
        cat2 = Categorical(list('acb'), categories=list('bca'), ordered=False)
        tm.assert_index_equal(cat2.categories, Index(['b', 'c', 'a']))
        assert not cat2.ordered
        cat3 = Categorical(list('acb'), ordered=True)
        tm.assert_index_equal(cat3.categories, Index(['a', 'b', 'c']))
        assert cat3.ordered
        cat4 = Categorical(list('acb'), categories=list('bca'), ordered=True)
        tm.assert_index_equal(cat4.categories, Index(['b', 'c', 'a']))
        assert cat4.ordered

    def test_set_ordered(self):
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        cat2 = cat.as_unordered()
        assert not cat2.ordered
        cat2 = cat.as_ordered()
        assert cat2.ordered
        assert cat2.set_ordered(True).ordered
        assert not cat2.set_ordered(False).ordered
        msg = "property 'ordered' of 'Categorical' object has no setter" if PY311 else "can't set attribute"
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = True
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = False

    def test_rename_categories(self):
        cat = Categorical(['a', 'b', 'c', 'a'])
        res = cat.rename_categories([1, 2, 3])
        tm.assert_numpy_array_equal(res.__array__(), np.array([1, 2, 3, 1], dtype=np.int64))
        tm.assert_index_equal(res.categories, Index([1, 2, 3]))
        exp_cat = np.array(['a', 'b', 'c', 'a'], dtype=np.object_)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)
        exp_cat = Index(['a', 'b', 'c'])
        tm.assert_index_equal(cat.categories, exp_cat)
        result = cat.rename_categories(lambda x: x.upper())
        expected = Categorical(['A', 'B', 'C', 'A'])
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize('new_categories', [[1, 2, 3, 4], [1, 2]])
    def test_rename_categories_wrong_length_raises(self, new_categories):
        cat = Categorical(['a', 'b', 'c', 'a'])
        msg = 'new categories need to have the same number of items as the old categories!'
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    def test_rename_categories_series(self):
        c = Categorical(['a', 'b'])
        result = c.rename_categories(Series([0, 1], index=['a', 'b']))
        expected = Categorical([0, 1])
        tm.assert_categorical_equal(result, expected)

    def test_rename_categories_dict(self):
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'a': 4, 'b': 3, 'c': 2, 'd': 1})
        expected = Index([4, 3, 2, 1])
        tm.assert_index_equal(res.categories, expected)
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'a': 1, 'c': 3})
        expected = Index([1, 'b', 3, 'd'])
        tm.assert_index_equal(res.categories, expected)
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6})
        expected = Index([1, 2, 3, 4])
        tm.assert_index_equal(res.categories, expected)
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'f': 1, 'g': 3})
        expected = Index(['a', 'b', 'c', 'd'])
        tm.assert_index_equal(res.categories, expected)

    def test_reorder_categories(self):
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        old = cat.copy()
        new = Categorical(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'], ordered=True)
        res = cat.reorder_categories(['c', 'b', 'a'])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize('new_categories', [['a'], ['a', 'b', 'd'], ['a', 'b', 'c', 'd']])
    def test_reorder_categories_raises(self, new_categories):
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        msg = 'items in new_categories are not the same as in old categories'
        with pytest.raises(ValueError, match=msg):
            cat.reorder_categories(new_categories)

    def test_add_categories(self):
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        old = cat.copy()
        new = Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c', 'd'], ordered=True)
        res = cat.add_categories('d')
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)
        res = cat.add_categories(['d'])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)
        cat = Categorical(list('abc'), ordered=True)
        expected = Categorical(list('abc'), categories=list('abcde'), ordered=True)
        res = cat.add_categories(Series(['d', 'e']))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(np.array(['d', 'e']))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(Index(['d', 'e']))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(['d', 'e'])
        tm.assert_categorical_equal(res, expected)

    def test_add_categories_existing_raises(self):
        cat = Categorical(['a', 'b', 'c', 'd'], ordered=True)
        msg = re.escape("new categories must not include old categories: {'d'}")
        with pytest.raises(ValueError, match=msg):
            cat.add_categories(['d'])

    def test_add_categories_losing_dtype_information(self):
        cat = Categorical(Series([1, 2], dtype='Int64'))
        ser = Series([4], dtype='Int64')
        result = cat.add_categories(ser)
        expected = Categorical(Series([1, 2], dtype='Int64'), categories=Series([1, 2, 4], dtype='Int64'))
        tm.assert_categorical_equal(result, expected)
        cat = Categorical(Series(['a', 'b', 'a'], dtype=StringDtype()))
        ser = Series(['d'], dtype=StringDtype())
        result = cat.add_categories(ser)
        expected = Categorical(Series(['a', 'b', 'a'], dtype=StringDtype()), categories=Series(['a', 'b', 'd'], dtype=StringDtype()))
        tm.assert_categorical_equal(result, expected)

    def test_set_categories(self):
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        exp_categories = Index(['c', 'b', 'a'])
        exp_values = np.array(['a', 'b', 'c', 'a'], dtype=np.object_)
        cat = cat.set_categories(['c', 'b', 'a'])
        res = cat.set_categories(['a', 'b', 'c'])
        tm.assert_index_equal(cat.categories, exp_categories)
        tm.assert_numpy_array_equal(cat.__array__(), exp_values)
        exp_categories_back = Index(['a', 'b', 'c'])
        tm.assert_index_equal(res.categories, exp_categories_back)
        tm.assert_numpy_array_equal(res.__array__(), exp_values)
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        res = cat.set_categories(['a'])
        tm.assert_numpy_array_equal(res.codes, np.array([0, -1, -1, 0], dtype=np.int8))
        res = cat.set_categories(['a', 'b', 'd'])
        tm.assert_numpy_array_equal(res.codes, np.array([0, 1, -1, 0], dtype=np.int8))
        tm.assert_index_equal(res.categories, Index(['a', 'b', 'd']))
        cat = cat.set_categories(['a', 'b', 'c', 'd'])
        exp_categories = Index(['a', 'b', 'c', 'd'])
        tm.assert_index_equal(cat.categories, exp_categories)
        c = Categorical([1, 2, 3, 4, 1], categories=[1, 2, 3, 4], ordered=True)
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, 2, 3, 0], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([1, 2, 3, 4]))
        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        c = c.set_categories([4, 3, 2, 1])
        tm.assert_numpy_array_equal(c._codes, np.array([3, 2, 1, 0, 3], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([4, 3, 2, 1]))
        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        assert c.min() == 4
        assert c.max() == 1
        c2 = c.set_categories([4, 3, 2, 1], ordered=False)
        assert not c2.ordered
        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))
        c2 = c.set_ordered(False).set_categories([4, 3, 2, 1])
        assert not c2.ordered
        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

    @pytest.mark.parametrize('values, categories, new_categories', [(['a', 'b', 'a'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'a'], ['a', 'b'], ['b', 'a']), (['b', 'a', 'a'], ['a', 'b'], ['a', 'b']), (['b', 'a', 'a'], ['a', 'b'], ['b', 'a']), (['a', 'b', 'c'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'c'], ['a', 'b'], ['b', 'a']), (['b', 'a', 'c'], ['a', 'b'], ['a', 'b']), (['b', 'a', 'c'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'c'], ['a', 'b'], ['a']), (['a', 'b', 'c'], ['a', 'b'], ['b']), (['b', 'a', 'c'], ['a', 'b'], ['a']), (['b', 'a', 'c'], ['a', 'b'], ['a']), (['a', 'b', 'c'], ['a', 'b'], ['d', 'e'])])
    @pytest.mark.parametrize('ordered', [True, False])
    def test_set_categories_many(self, values, categories, new_categories, ordered):
        c = Categorical(values, categories)
        expected = Categorical(values, new_categories, ordered)
        result = c.set_categories(new_categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_rename_less(self):
        cat = Categorical(['A', 'B'])
        result = cat.set_categories(['A'], rename=True)
        expected = Categorical(['A', np.nan])
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_private(self):
        cat = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c', 'd'])
        cat._set_categories(['a', 'c', 'd', 'e'])
        expected = Categorical(['a', 'c', 'd'], categories=list('acde'))
        tm.assert_categorical_equal(cat, expected)
        cat = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c', 'd'])
        cat._set_categories(['a', 'c', 'd', 'e'], fastpath=True)
        expected = Categorical(['a', 'c', 'd'], categories=list('acde'))
        tm.assert_categorical_equal(cat, expected)

    def test_remove_categories(self):
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        old = cat.copy()
        new = Categorical(['a', 'b', np.nan, 'a'], categories=['a', 'b'], ordered=True)
        res = cat.remove_categories('c')
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)
        res = cat.remove_categories(['c'])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize('removals', [['c'], ['c', np.nan], 'c', ['c', 'c']])
    def test_remove_categories_raises(self, removals):
        cat = Categorical(['a', 'b', 'a'])
        message = re.escape("removals must all be in old categories: {'c'}")
        with pytest.raises(ValueError, match=message):
            cat.remove_categories(removals)

    def test_remove_unused_categories(self):
        c = Categorical(['a', 'b', 'c', 'd', 'a'], categories=['a', 'b', 'c', 'd', 'e'])
        exp_categories_all = Index(['a', 'b', 'c', 'd', 'e'])
        exp_categories_dropped = Index(['a', 'b', 'c', 'd'])
        tm.assert_index_equal(c.categories, exp_categories_all)
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, exp_categories_dropped)
        tm.assert_index_equal(c.categories, exp_categories_all)
        c = Categorical(['a', 'b', 'c', np.nan], categories=['a', 'b', 'c', 'd', 'e'])
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, Index(np.array(['a', 'b', 'c'])))
        exp_codes = np.array([0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(res.codes, exp_codes)
        tm.assert_index_equal(c.categories, exp_categories_all)
        val = ['F', np.nan, 'D', 'B', 'D', 'F', np.nan]
        cat = Categorical(values=val, categories=list('ABCDEFG'))
        out = cat.remove_unused_categories()
        tm.assert_index_equal(out.categories, Index(['B', 'D', 'F']))
        exp_codes = np.array([2, -1, 1, 0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(out.codes, exp_codes)
        assert out.tolist() == val
        alpha = list('abcdefghijklmnopqrstuvwxyz')
        val = np.random.default_rng(2).choice(alpha[::2], 10000).astype('object')
        val[np.random.default_rng(2).choice(len(val), 100)] = np.nan
        cat = Categorical(values=val, categories=alpha)
        out = cat.remove_unused_categories()
        assert out.tolist() == val.tolist()