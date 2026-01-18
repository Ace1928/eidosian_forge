import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameSortIndexKey:

    def test_sort_multi_index_key(self):
        df = DataFrame({'a': [3, 1, 2], 'b': [0, 0, 0], 'c': [0, 1, 2], 'd': list('abc')}).set_index(list('abc'))
        result = df.sort_index(level=list('ac'), key=lambda x: x)
        expected = DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0], 'c': [1, 2, 0], 'd': list('bca')}).set_index(list('abc'))
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(level=list('ac'), key=lambda x: -x)
        expected = DataFrame({'a': [3, 2, 1], 'b': [0, 0, 0], 'c': [0, 2, 1], 'd': list('acb')}).set_index(list('abc'))
        tm.assert_frame_equal(result, expected)

    def test_sort_index_key(self):
        df = DataFrame(np.arange(6, dtype='int64'), index=list('aaBBca'))
        result = df.sort_index()
        expected = df.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(key=lambda x: x.str.lower())
        expected = df.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = df.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_key_int(self):
        df = DataFrame(np.arange(6, dtype='int64'), index=np.arange(6, dtype='int64'))
        result = df.sort_index()
        tm.assert_frame_equal(result, df)
        result = df.sort_index(key=lambda x: -x)
        expected = df.sort_index(ascending=False)
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(key=lambda x: 2 * x)
        tm.assert_frame_equal(result, df)

    def test_sort_multi_index_key_str(self):
        df = DataFrame({'a': ['B', 'a', 'C'], 'b': [0, 1, 0], 'c': list('abc'), 'd': [0, 1, 2]}).set_index(list('abc'))
        result = df.sort_index(level='a', key=lambda x: x.str.lower())
        expected = DataFrame({'a': ['a', 'B', 'C'], 'b': [1, 0, 0], 'c': list('bac'), 'd': [1, 0, 2]}).set_index(list('abc'))
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(level=list('abc'), key=lambda x: x.str.lower() if x.name in ['a', 'c'] else -x)
        expected = DataFrame({'a': ['a', 'B', 'C'], 'b': [1, 0, 0], 'c': list('bac'), 'd': [1, 0, 2]}).set_index(list('abc'))
        tm.assert_frame_equal(result, expected)

    def test_changes_length_raises(self):
        df = DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError, match='change the shape'):
            df.sort_index(key=lambda x: x[:1])

    def test_sort_index_multiindex_sparse_column(self):
        expected = DataFrame({i: pd.array([0.0, 0.0, 0.0, 0.0], dtype=pd.SparseDtype('float64', 0.0)) for i in range(4)}, index=MultiIndex.from_product([[1, 2], [1, 2]]))
        result = expected.sort_index(level=0)
        tm.assert_frame_equal(result, expected)

    def test_sort_index_na_position(self):
        df = DataFrame([1, 2], index=MultiIndex.from_tuples([(1, 1), (1, pd.NA)]))
        expected = df.copy()
        result = df.sort_index(level=[0, 1], na_position='last')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_index_multiindex_sort_remaining(self, ascending):
        df = DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}, index=MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y'), ('c', 'x')]))
        result = df.sort_index(level=1, sort_remaining=False, ascending=ascending)
        if ascending:
            expected = DataFrame({'A': [1, 3, 5, 2, 4], 'B': [10, 30, 50, 20, 40]}, index=MultiIndex.from_tuples([('a', 'x'), ('b', 'x'), ('c', 'x'), ('a', 'y'), ('b', 'y')]))
        else:
            expected = DataFrame({'A': [2, 4, 1, 3, 5], 'B': [20, 40, 10, 30, 50]}, index=MultiIndex.from_tuples([('a', 'y'), ('b', 'y'), ('a', 'x'), ('b', 'x'), ('c', 'x')]))
        tm.assert_frame_equal(result, expected)