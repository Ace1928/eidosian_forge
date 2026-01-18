from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestMultiIndexConcat:

    def test_concat_multiindex_with_keys(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        index = frame.index
        result = concat([frame, frame], keys=[0, 1], names=['iteration'])
        assert result.index.names == ('iteration',) + index.names
        tm.assert_frame_equal(result.loc[0], frame)
        tm.assert_frame_equal(result.loc[1], frame)
        assert result.index.nlevels == 3

    def test_concat_multiindex_with_none_in_index_names(self):
        index = MultiIndex.from_product([[1], range(5)], names=['level1', None])
        df = DataFrame({'col': range(5)}, index=index, dtype=np.int32)
        result = concat([df, df], keys=[1, 2], names=['level2'])
        index = MultiIndex.from_product([[1, 2], [1], range(5)], names=['level2', 'level1', None])
        expected = DataFrame({'col': list(range(5)) * 2}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)
        result = concat([df, df[:2]], keys=[1, 2], names=['level2'])
        level2 = [1] * 5 + [2] * 2
        level1 = [1] * 7
        no_name = list(range(5)) + list(range(2))
        tuples = list(zip(level2, level1, no_name))
        index = MultiIndex.from_tuples(tuples, names=['level2', 'level1', None])
        expected = DataFrame({'col': no_name}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_rangeindex(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)))
        df.index = MultiIndex(levels=[pd.RangeIndex(3), pd.RangeIndex(3)], codes=[np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)])
        res = concat([df.iloc[[2, 3, 4], :], df.iloc[[5], :]])
        exp = df.iloc[[2, 3, 4, 5], :]
        tm.assert_frame_equal(res, exp)

    def test_concat_multiindex_dfs_with_deepcopy(self):
        example_multiindex1 = MultiIndex.from_product([['a'], ['b']])
        example_dataframe1 = DataFrame([0], index=example_multiindex1)
        example_multiindex2 = MultiIndex.from_product([['a'], ['c']])
        example_dataframe2 = DataFrame([1], index=example_multiindex2)
        example_dict = {'s1': example_dataframe1, 's2': example_dataframe2}
        expected_index = MultiIndex(levels=[['s1', 's2'], ['a'], ['b', 'c']], codes=[[0, 1], [0, 0], [0, 1]], names=['testname', None, None])
        expected = DataFrame([[0], [1]], index=expected_index)
        result_copy = concat(deepcopy(example_dict), names=['testname'])
        tm.assert_frame_equal(result_copy, expected)
        result_no_copy = concat(example_dict, names=['testname'])
        tm.assert_frame_equal(result_no_copy, expected)

    @pytest.mark.parametrize('mi1_list', [[['a'], range(2)], [['b'], np.arange(2.0, 4.0)], [['c'], ['A', 'B']], [['d'], pd.date_range(start='2017', end='2018', periods=2)]])
    @pytest.mark.parametrize('mi2_list', [[['a'], range(2)], [['b'], np.arange(2.0, 4.0)], [['c'], ['A', 'B']], [['d'], pd.date_range(start='2017', end='2018', periods=2)]])
    def test_concat_with_various_multiindex_dtypes(self, mi1_list: list, mi2_list: list):
        mi1 = MultiIndex.from_product(mi1_list)
        mi2 = MultiIndex.from_product(mi2_list)
        df1 = DataFrame(np.zeros((1, len(mi1))), columns=mi1)
        df2 = DataFrame(np.zeros((1, len(mi2))), columns=mi2)
        if mi1_list[0] == mi2_list[0]:
            expected_mi = MultiIndex(levels=[mi1_list[0], list(mi1_list[1])], codes=[[0, 0, 0, 0], [0, 1, 0, 1]])
        else:
            expected_mi = MultiIndex(levels=[mi1_list[0] + mi2_list[0], list(mi1_list[1]) + list(mi2_list[1])], codes=[[0, 0, 1, 1], [0, 1, 2, 3]])
        expected_df = DataFrame(np.zeros((1, len(expected_mi))), columns=expected_mi)
        with tm.assert_produces_warning(None):
            result_df = concat((df1, df2), axis=1)
        tm.assert_frame_equal(expected_df, result_df)

    def test_concat_multiindex_(self):
        df = DataFrame({'col': ['a', 'b', 'c']}, index=['1', '2', '2'])
        df = concat([df], keys=['X'])
        iterables = [['X'], ['1', '2', '2']]
        result_index = df.index
        expected_index = MultiIndex.from_product(iterables)
        tm.assert_index_equal(result_index, expected_index)
        result_df = df
        expected_df = DataFrame({'col': ['a', 'b', 'c']}, index=MultiIndex.from_product(iterables))
        tm.assert_frame_equal(result_df, expected_df)

    def test_concat_with_key_not_unique(self):
        df1 = DataFrame({'name': [1]})
        df2 = DataFrame({'name': [2]})
        df3 = DataFrame({'name': [3]})
        df_a = concat([df1, df2, df3], keys=['x', 'y', 'x'])
        with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
            out_a = df_a.loc[('x', 0), :]
        df_b = DataFrame({'name': [1, 2, 3]}, index=Index([('x', 0), ('y', 0), ('x', 0)]))
        with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
            out_b = df_b.loc['x', 0]
        tm.assert_frame_equal(out_a, out_b)
        df1 = DataFrame({'name': ['a', 'a', 'b']})
        df2 = DataFrame({'name': ['a', 'b']})
        df3 = DataFrame({'name': ['c', 'd']})
        df_a = concat([df1, df2, df3], keys=['x', 'y', 'x'])
        with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
            out_a = df_a.loc[('x', 0), :]
        df_b = DataFrame({'a': ['x', 'x', 'x', 'y', 'y', 'x', 'x'], 'b': [0, 1, 2, 0, 1, 0, 1], 'name': list('aababcd')}).set_index(['a', 'b'])
        df_b.index.names = [None, None]
        with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
            out_b = df_b.loc[('x', 0), :]
        tm.assert_frame_equal(out_a, out_b)

    def test_concat_with_duplicated_levels(self):
        df1 = DataFrame({'A': [1]}, index=['x'])
        df2 = DataFrame({'A': [1]}, index=['y'])
        msg = "Level values not unique: \\['x', 'y', 'y'\\]"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], keys=['x', 'y'], levels=[['x', 'y', 'y']])

    @pytest.mark.parametrize('levels', [[['x', 'y']], [['x', 'y', 'y']]])
    def test_concat_with_levels_with_none_keys(self, levels):
        df1 = DataFrame({'A': [1]}, index=['x'])
        df2 = DataFrame({'A': [1]}, index=['y'])
        msg = 'levels supported only when keys is not None'
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], levels=levels)

    def test_concat_range_index_result(self):
        df1 = DataFrame({'a': [1, 2]})
        df2 = DataFrame({'b': [1, 2]})
        result = concat([df1, df2], sort=True, axis=1)
        expected = DataFrame({'a': [1, 2], 'b': [1, 2]})
        tm.assert_frame_equal(result, expected)
        expected_index = pd.RangeIndex(0, 2)
        tm.assert_index_equal(result.index, expected_index, exact=True)

    def test_concat_index_keep_dtype(self):
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype='object'))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype='object'))
        result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
        expected = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype='object'))
        tm.assert_frame_equal(result, expected)

    def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype):
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype=any_numeric_ea_dtype))
        result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
        expected = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['Int8', 'Int16', 'Int32'])
    def test_concat_index_find_common(self, dtype):
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=dtype))
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype='Int32'))
        result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
        expected = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype='Int32'))
        tm.assert_frame_equal(result, expected)

    def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string):
        s1 = Series(['a', 'b', 'c'])
        s2 = Series(['a', 'b'])
        s3 = Series(['a', 'b', 'c', 'd'])
        s4 = Series([], dtype=object if not using_infer_string else 'string[pyarrow_numpy]')
        result = concat([s1, s2, s3, s4], sort=False, join='outer', ignore_index=False, axis=1)
        expected = DataFrame([['a'] * 3 + [np.nan], ['b'] * 3 + [np.nan], ['c', np.nan] * 2, [np.nan] * 2 + ['d'] + [np.nan]], dtype=object if not using_infer_string else 'string[pyarrow_numpy]')
        tm.assert_frame_equal(result, expected, check_index_type=True, check_column_type=True)