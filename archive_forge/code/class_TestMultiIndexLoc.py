import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestMultiIndexLoc:

    def test_loc_setitem_frame_with_multiindex(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        frame.loc[('bar', 'two'), 'B'] = 5
        assert frame.loc[('bar', 'two'), 'B'] == 5
        df = frame.copy()
        df.columns = list(range(3))
        df.loc[('bar', 'two'), 1] = 7
        assert df.loc[('bar', 'two'), 1] == 7

    def test_loc_getitem_general(self, any_real_numpy_dtype):
        dtype = any_real_numpy_dtype
        data = {'amount': {0: 700, 1: 600, 2: 222, 3: 333, 4: 444}, 'col': {0: 3.5, 1: 3.5, 2: 4.0, 3: 4.0, 4: 4.0}, 'num': {0: 12, 1: 11, 2: 12, 3: 12, 4: 12}}
        df = DataFrame(data)
        df = df.astype({'col': dtype, 'num': dtype})
        df = df.set_index(keys=['col', 'num'])
        key = (4.0, 12)
        with tm.assert_produces_warning(PerformanceWarning):
            tm.assert_frame_equal(df.loc[key], df.iloc[2:])
        return_value = df.sort_index(inplace=True)
        assert return_value is None
        res = df.loc[key]
        col_arr = np.array([4.0] * 3, dtype=dtype)
        year_arr = np.array([12] * 3, dtype=dtype)
        index = MultiIndex.from_arrays([col_arr, year_arr], names=['col', 'num'])
        expected = DataFrame({'amount': [222, 333, 444]}, index=index)
        tm.assert_frame_equal(res, expected)

    def test_loc_getitem_multiindex_missing_label_raises(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
        with pytest.raises(KeyError, match='^2$'):
            df.loc[2]

    def test_loc_getitem_list_of_tuples_with_multiindex(self, multiindex_year_month_day_dataframe_random_data):
        ser = multiindex_year_month_day_dataframe_random_data['A']
        expected = ser.reindex(ser.index[49:51])
        result = ser.loc[[(2000, 3, 10), (2000, 3, 13)]]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_series(self):
        index = MultiIndex.from_product([[1, 2, 3], ['A', 'B', 'C']])
        x = Series(index=index, data=range(9), dtype=np.float64)
        y = Series([1, 3])
        expected = Series(data=[0, 1, 2, 6, 7, 8], index=MultiIndex.from_product([[1, 3], ['A', 'B', 'C']]), dtype=np.float64)
        result = x.loc[y]
        tm.assert_series_equal(result, expected)
        result = x.loc[[1, 3]]
        tm.assert_series_equal(result, expected)
        y1 = Series([1, 3], index=[1, 2])
        result = x.loc[y1]
        tm.assert_series_equal(result, expected)
        empty = Series(data=[], dtype=np.float64)
        expected = Series([], index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64), dtype=np.float64)
        result = x.loc[empty]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_array(self):
        index = MultiIndex.from_product([[1, 2, 3], ['A', 'B', 'C']])
        x = Series(index=index, data=range(9), dtype=np.float64)
        y = np.array([1, 3])
        expected = Series(data=[0, 1, 2, 6, 7, 8], index=MultiIndex.from_product([[1, 3], ['A', 'B', 'C']]), dtype=np.float64)
        result = x.loc[y]
        tm.assert_series_equal(result, expected)
        empty = np.array([])
        expected = Series([], index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64), dtype='float64')
        result = x.loc[empty]
        tm.assert_series_equal(result, expected)
        scalar = np.int64(1)
        expected = Series(data=[0, 1, 2], index=['A', 'B', 'C'], dtype=np.float64)
        result = x.loc[scalar]
        tm.assert_series_equal(result, expected)

    def test_loc_multiindex_labels(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[['i', 'i', 'j'], ['A', 'A', 'B']], index=[['i', 'i', 'j'], ['X', 'X', 'Y']])
        expected = df.iloc[[0, 1]].droplevel(0)
        result = df.loc['i']
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[:, [2]].droplevel(0, axis=1)
        result = df.loc[:, 'j']
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[[2], [2]].droplevel(0).droplevel(0, axis=1)
        result = df.loc['j'].loc[:, 'j']
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[[0, 1]]
        result = df.loc['i', 'X']
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_ints(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
        expected = df.iloc[[0, 1]].droplevel(0)
        result = df.loc[4]
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_missing_label_raises(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
        with pytest.raises(KeyError, match='^2$'):
            df.loc[2]

    @pytest.mark.parametrize('key, pos', [([2, 4], [0, 1]), ([2], []), ([2, 3], [])])
    def test_loc_multiindex_list_missing_label(self, key, pos):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
        with pytest.raises(KeyError, match='not in index'):
            df.loc[key]

    def test_loc_multiindex_too_many_dims_raises(self):
        s = Series(range(8), index=MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']]))
        with pytest.raises(KeyError, match="^\\('a', 'b'\\)$"):
            s.loc['a', 'b']
        with pytest.raises(KeyError, match="^\\('a', 'd', 'g'\\)$"):
            s.loc['a', 'd', 'g']
        with pytest.raises(IndexingError, match='Too many indexers'):
            s.loc['a', 'd', 'g', 'j']

    def test_loc_multiindex_indexer_none(self):
        attributes = ['Attribute' + str(i) for i in range(1)]
        attribute_values = ['Value' + str(i) for i in range(5)]
        index = MultiIndex.from_product([attributes, attribute_values])
        df = 0.1 * np.random.default_rng(2).standard_normal((10, 1 * 5)) + 0.5
        df = DataFrame(df, columns=index)
        result = df[attributes]
        tm.assert_frame_equal(result, df)
        df = DataFrame(np.arange(12).reshape(-1, 1), index=MultiIndex.from_product([[1, 2, 3, 4], [1, 2, 3]]))
        expected = df.loc[([1, 2],), :]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_incomplete(self):
        s = Series(np.arange(15, dtype='int64'), MultiIndex.from_product([range(5), ['a', 'b', 'c']]))
        expected = s.loc[:, 'a':'c']
        result = s.loc[0:4, 'a':'c']
        tm.assert_series_equal(result, expected)
        result = s.loc[:4, 'a':'c']
        tm.assert_series_equal(result, expected)
        result = s.loc[0:, 'a':'c']
        tm.assert_series_equal(result, expected)
        s = Series(np.arange(15, dtype='int64'), MultiIndex.from_product([range(5), ['a', 'b', 'c']]))
        expected = s.iloc[[6, 7, 8, 12, 13, 14]]
        result = s.loc[2:4:2, 'a':'c']
        tm.assert_series_equal(result, expected)

    def test_get_loc_single_level(self, single_level_multiindex):
        single_level = single_level_multiindex
        s = Series(np.random.default_rng(2).standard_normal(len(single_level)), index=single_level)
        for k in single_level.values:
            s[k]

    def test_loc_getitem_int_slice(self):
        index = MultiIndex.from_product([[6, 7, 8], ['a', 'b']])
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 6)), index, index)
        result = df.loc[6:8, :]
        expected = df
        tm.assert_frame_equal(result, expected)
        index = MultiIndex.from_product([[10, 20, 30], ['a', 'b']])
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 6)), index, index)
        result = df.loc[20:30, :]
        expected = df.iloc[2:]
        tm.assert_frame_equal(result, expected)
        result = df.loc[10, :]
        expected = df.iloc[0:2]
        expected.index = ['a', 'b']
        tm.assert_frame_equal(result, expected)
        result = df.loc[:, 10]
        expected = df[10]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexer_type_1', (list, tuple, set, slice, np.ndarray, Series, Index))
    @pytest.mark.parametrize('indexer_type_2', (list, tuple, set, slice, np.ndarray, Series, Index))
    def test_loc_getitem_nested_indexer(self, indexer_type_1, indexer_type_2):

        def convert_nested_indexer(indexer_type, keys):
            if indexer_type == np.ndarray:
                return np.array(keys)
            if indexer_type == slice:
                return slice(*keys)
            return indexer_type(keys)
        a = [10, 20, 30]
        b = [1, 2, 3]
        index = MultiIndex.from_product([a, b])
        df = DataFrame(np.arange(len(index), dtype='int64'), index=index, columns=['Data'])
        keys = ([10, 20], [2, 3])
        types = (indexer_type_1, indexer_type_2)
        indexer = tuple((convert_nested_indexer(indexer_type, k) for indexer_type, k in zip(types, keys)))
        if indexer_type_1 is set or indexer_type_2 is set:
            with pytest.raises(TypeError, match='as an indexer is not supported'):
                df.loc[indexer, 'Data']
            return
        else:
            result = df.loc[indexer, 'Data']
        expected = Series([1, 2, 4, 5], name='Data', index=MultiIndex.from_product(keys))
        tm.assert_series_equal(result, expected)

    def test_multiindex_loc_one_dimensional_tuple(self, frame_or_series):
        mi = MultiIndex.from_tuples([('a', 'A'), ('b', 'A')])
        obj = frame_or_series([1, 2], index=mi)
        obj.loc['a',] = 0
        expected = frame_or_series([0, 2], index=mi)
        tm.assert_equal(obj, expected)

    @pytest.mark.parametrize('indexer', [('a',), 'a'])
    def test_multiindex_one_dimensional_tuple_columns(self, indexer):
        mi = MultiIndex.from_tuples([('a', 'A'), ('b', 'A')])
        obj = DataFrame([1, 2], index=mi)
        obj.loc[indexer, :] = 0
        expected = DataFrame([0, 2], index=mi)
        tm.assert_frame_equal(obj, expected)

    @pytest.mark.parametrize('indexer, exp_value', [(slice(None), 1.0), ((1, 2), np.nan)])
    def test_multiindex_setitem_columns_enlarging(self, indexer, exp_value):
        mi = MultiIndex.from_tuples([(1, 2), (3, 4)])
        df = DataFrame([[1, 2], [3, 4]], index=mi, columns=['a', 'b'])
        df.loc[indexer, ['c', 'd']] = 1.0
        expected = DataFrame([[1, 2, 1.0, 1.0], [3, 4, exp_value, exp_value]], index=mi, columns=['a', 'b', 'c', 'd'])
        tm.assert_frame_equal(df, expected)

    def test_sorted_multiindex_after_union(self):
        midx = MultiIndex.from_product([pd.date_range('20110101', periods=2), Index(['a', 'b'])])
        ser1 = Series(1, index=midx)
        ser2 = Series(1, index=midx[:2])
        df = pd.concat([ser1, ser2], axis=1)
        expected = df.copy()
        result = df.loc['2011-01-01':'2011-01-02']
        tm.assert_frame_equal(result, expected)
        df = DataFrame({0: ser1, 1: ser2})
        result = df.loc['2011-01-01':'2011-01-02']
        tm.assert_frame_equal(result, expected)
        df = pd.concat([ser1, ser2.reindex(ser1.index)], axis=1)
        result = df.loc['2011-01-01':'2011-01-02']
        tm.assert_frame_equal(result, expected)

    def test_loc_no_second_level_index(self):
        df = DataFrame(index=MultiIndex.from_product([list('ab'), list('cd'), list('e')]), columns=['Val'])
        res = df.loc[np.s_[:, 'c', :]]
        expected = DataFrame(index=MultiIndex.from_product([list('ab'), list('e')]), columns=['Val'])
        tm.assert_frame_equal(res, expected)

    def test_loc_multi_index_key_error(self):
        df = DataFrame({(1, 2): ['a', 'b', 'c'], (1, 3): ['d', 'e', 'f'], (2, 2): ['g', 'h', 'i'], (2, 4): ['j', 'k', 'l']})
        with pytest.raises(KeyError, match='(1, 4)'):
            df.loc[0, (1, 4)]