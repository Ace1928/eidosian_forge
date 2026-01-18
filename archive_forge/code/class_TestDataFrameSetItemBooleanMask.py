from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
class TestDataFrameSetItemBooleanMask:

    @td.skip_array_manager_invalid_test
    @pytest.mark.parametrize('mask_type', [lambda df: df > np.abs(df) / 2, lambda df: (df > np.abs(df) / 2).values], ids=['dataframe', 'array'])
    def test_setitem_boolean_mask(self, mask_type, float_frame):
        df = float_frame.copy()
        mask = mask_type(df)
        result = df.copy()
        result[mask] = np.nan
        expected = df.values.copy()
        expected[np.array(mask)] = np.nan
        expected = DataFrame(expected, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(reason='Currently empty indexers are treated as all False')
    @pytest.mark.parametrize('box', [list, np.array, Series])
    def test_setitem_loc_empty_indexer_raises_with_non_empty_value(self, box):
        df = DataFrame({'a': ['a'], 'b': [1], 'c': [1]})
        if box == Series:
            indexer = box([], dtype='object')
        else:
            indexer = box([])
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df.loc[indexer, ['b']] = [1]

    @pytest.mark.parametrize('box', [list, np.array, Series])
    def test_setitem_loc_only_false_indexer_dtype_changed(self, box):
        df = DataFrame({'a': ['a'], 'b': [1], 'c': [1]})
        indexer = box([False])
        df.loc[indexer, ['b']] = 10 - df['c']
        expected = DataFrame({'a': ['a'], 'b': [1], 'c': [1]})
        tm.assert_frame_equal(df, expected)
        df.loc[indexer, ['b']] = 9
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [tm.setitem, tm.loc])
    def test_setitem_boolean_mask_aligning(self, indexer):
        df = DataFrame({'a': [1, 4, 2, 3], 'b': [5, 6, 7, 8]})
        expected = df.copy()
        mask = df['a'] >= 3
        indexer(df)[mask] = indexer(df)[mask].sort_values('a')
        tm.assert_frame_equal(df, expected)

    def test_setitem_mask_categorical(self):
        cats2 = Categorical(['a', 'a', 'b', 'b', 'a', 'a', 'a'], categories=['a', 'b'])
        idx2 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({'cats': cats2, 'values': values2}, index=idx2)
        catsf = Categorical(['a', 'a', 'c', 'c', 'a', 'a', 'a'], categories=['a', 'b', 'c'])
        idxf = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        valuesf = [1, 1, 3, 3, 1, 1, 1]
        df = DataFrame({'cats': catsf, 'values': valuesf}, index=idxf)
        exp_fancy = exp_multi_row.copy()
        exp_fancy['cats'] = exp_fancy['cats'].cat.set_categories(['a', 'b', 'c'])
        mask = df['cats'] == 'c'
        df[mask] = ['b', 2]
        tm.assert_frame_equal(df, exp_fancy)

    @pytest.mark.parametrize('dtype', ['float', 'int64'])
    @pytest.mark.parametrize('kwargs', [{}, {'index': [1]}, {'columns': ['A']}])
    def test_setitem_empty_frame_with_boolean(self, dtype, kwargs):
        kwargs['dtype'] = dtype
        df = DataFrame(**kwargs)
        df2 = df.copy()
        df[df > df2] = 47
        tm.assert_frame_equal(df, df2)

    def test_setitem_boolean_indexing(self):
        idx = list(range(3))
        cols = ['A', 'B', 'C']
        df1 = DataFrame(index=idx, columns=cols, data=np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]], dtype=float))
        df2 = DataFrame(index=idx, columns=cols, data=np.ones((len(idx), len(cols))))
        expected = DataFrame(index=idx, columns=cols, data=np.array([[0.0, 0.5, 1.0], [1.5, 2.0, -1], [-1, -1, -1]], dtype=float))
        df1[df1 > 2.0 * df2] = -1
        tm.assert_frame_equal(df1, expected)
        with pytest.raises(ValueError, match='Item wrong length'):
            df1[df1.index[:-1] > 2] = -1

    def test_loc_setitem_all_false_boolean_two_blocks(self):
        df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': 'a'})
        expected = df.copy()
        indexer = Series([False, False], name='c')
        df.loc[indexer, ['b']] = DataFrame({'b': [5, 6]}, index=[0, 1])
        tm.assert_frame_equal(df, expected)

    def test_setitem_ea_boolean_mask(self):
        df = DataFrame([[-1, 2], [3, -4]])
        expected = DataFrame([[0, 2], [3, 0]])
        boolean_indexer = DataFrame({0: Series([True, False], dtype='boolean'), 1: Series([pd.NA, True], dtype='boolean')})
        df[boolean_indexer] = 0
        tm.assert_frame_equal(df, expected)