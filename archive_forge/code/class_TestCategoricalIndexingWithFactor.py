import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
class TestCategoricalIndexingWithFactor:

    def test_getitem(self):
        factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
        assert factor[0] == 'a'
        assert factor[-1] == 'c'
        subf = factor[[0, 1, 2]]
        tm.assert_numpy_array_equal(subf._codes, np.array([0, 1, 1], dtype=np.int8))
        subf = factor[np.asarray(factor) == 'c']
        tm.assert_numpy_array_equal(subf._codes, np.array([2, 2, 2], dtype=np.int8))

    def test_setitem(self):
        factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
        c = factor.copy()
        c[0] = 'b'
        assert c[0] == 'b'
        c[-1] = 'a'
        assert c[-1] == 'a'
        c = factor.copy()
        indexer = np.zeros(len(c), dtype='bool')
        indexer[0] = True
        indexer[-1] = True
        c[indexer] = 'c'
        expected = Categorical(['c', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
        tm.assert_categorical_equal(c, expected)

    @pytest.mark.parametrize('other', [Categorical(['b', 'a']), Categorical(['b', 'a'], categories=['b', 'a'])])
    def test_setitem_same_but_unordered(self, other):
        target = Categorical(['a', 'b'], categories=['a', 'b'])
        mask = np.array([True, False])
        target[mask] = other[mask]
        expected = Categorical(['b', 'b'], categories=['a', 'b'])
        tm.assert_categorical_equal(target, expected)

    @pytest.mark.parametrize('other', [Categorical(['b', 'a'], categories=['b', 'a', 'c']), Categorical(['b', 'a'], categories=['a', 'b', 'c']), Categorical(['a', 'a'], categories=['a']), Categorical(['b', 'b'], categories=['b'])])
    def test_setitem_different_unordered_raises(self, other):
        target = Categorical(['a', 'b'], categories=['a', 'b'])
        mask = np.array([True, False])
        msg = 'Cannot set a Categorical with another, without identical categories'
        with pytest.raises(TypeError, match=msg):
            target[mask] = other[mask]

    @pytest.mark.parametrize('other', [Categorical(['b', 'a']), Categorical(['b', 'a'], categories=['b', 'a'], ordered=True), Categorical(['b', 'a'], categories=['a', 'b', 'c'], ordered=True)])
    def test_setitem_same_ordered_raises(self, other):
        target = Categorical(['a', 'b'], categories=['a', 'b'], ordered=True)
        mask = np.array([True, False])
        msg = 'Cannot set a Categorical with another, without identical categories'
        with pytest.raises(TypeError, match=msg):
            target[mask] = other[mask]

    def test_setitem_tuple(self):
        cat = Categorical([(0, 1), (0, 2), (0, 1)])
        cat[1] = cat[0]
        assert cat[1] == (0, 1)

    def test_setitem_listlike(self):
        cat = Categorical(np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8)).add_categories([-1000])
        indexer = np.array([100000]).astype(np.int64)
        cat[indexer] = -1000
        result = cat.codes[np.array([100000]).astype(np.int64)]
        tm.assert_numpy_array_equal(result, np.array([5], dtype='int8'))