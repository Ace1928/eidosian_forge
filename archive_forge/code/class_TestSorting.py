from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
class TestSorting:

    @pytest.mark.slow
    def test_int64_overflow(self):
        B = np.concatenate((np.arange(1000), np.arange(1000), np.arange(500)))
        A = np.arange(2500)
        df = DataFrame({'A': A, 'B': B, 'C': A, 'D': B, 'E': A, 'F': B, 'G': A, 'H': B, 'values': np.random.default_rng(2).standard_normal(2500)})
        lg = df.groupby(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        rg = df.groupby(['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'])
        left = lg.sum()['values']
        right = rg.sum()['values']
        exp_index, _ = left.index.sortlevel()
        tm.assert_index_equal(left.index, exp_index)
        exp_index, _ = right.index.sortlevel(0)
        tm.assert_index_equal(right.index, exp_index)
        tups = list(map(tuple, df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].values))
        tups = com.asarray_tuplesafe(tups)
        expected = df.groupby(tups).sum()['values']
        for k, v in expected.items():
            assert left[k] == right[k[::-1]]
            assert left[k] == v
        assert len(left) == len(right)

    def test_int64_overflow_groupby_large_range(self):
        values = range(55109)
        data = DataFrame.from_dict({'a': values, 'b': values, 'c': values, 'd': values})
        grouped = data.groupby(['a', 'b', 'c', 'd'])
        assert len(grouped) == len(values)

    @pytest.mark.parametrize('agg', ['mean', 'median'])
    def test_int64_overflow_groupby_large_df_shuffled(self, agg):
        rs = np.random.default_rng(2)
        arr = rs.integers(-1 << 12, 1 << 12, (1 << 15, 5))
        i = rs.choice(len(arr), len(arr) * 4)
        arr = np.vstack((arr, arr[i]))
        i = rs.permutation(len(arr))
        arr = arr[i]
        df = DataFrame(arr, columns=list('abcde'))
        df['jim'], df['joe'] = np.zeros((2, len(df)))
        gr = df.groupby(list('abcde'))
        assert is_int64_overflow_possible(gr._grouper.shape)
        mi = MultiIndex.from_arrays([ar.ravel() for ar in np.array_split(np.unique(arr, axis=0), 5, axis=1)], names=list('abcde'))
        res = DataFrame(np.zeros((len(mi), 2)), columns=['jim', 'joe'], index=mi).sort_index()
        tm.assert_frame_equal(getattr(gr, agg)(), res)

    @pytest.mark.parametrize('order, na_position, exp', [[True, 'last', list(range(5, 105)) + list(range(5)) + list(range(105, 110))], [True, 'first', list(range(5)) + list(range(105, 110)) + list(range(5, 105))], [False, 'last', list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))], [False, 'first', list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))]])
    def test_lexsort_indexer(self, order, na_position, exp):
        keys = [[np.nan] * 5 + list(range(100)) + [np.nan] * 5]
        result = lexsort_indexer(keys, orders=order, na_position=na_position)
        tm.assert_numpy_array_equal(result, np.array(exp, dtype=np.intp))

    @pytest.mark.parametrize('ascending, na_position, exp', [[True, 'last', list(range(5, 105)) + list(range(5)) + list(range(105, 110))], [True, 'first', list(range(5)) + list(range(105, 110)) + list(range(5, 105))], [False, 'last', list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))], [False, 'first', list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))]])
    def test_nargsort(self, ascending, na_position, exp):
        items = np.array([np.nan] * 5 + list(range(100)) + [np.nan] * 5, dtype='O')
        result = nargsort(items, kind='mergesort', ascending=ascending, na_position=na_position)
        tm.assert_numpy_array_equal(result, np.array(exp), check_dtype=False)