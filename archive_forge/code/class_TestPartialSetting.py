import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestPartialSetting:

    def test_partial_setting(self):
        s_orig = Series([1, 2, 3])
        s = s_orig.copy()
        s[5] = 5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s = s_orig.copy()
        s.loc[5] = 5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s = s_orig.copy()
        s[5] = 5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s = s_orig.copy()
        s.loc[5] = 5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s = s_orig.copy()
        msg = 'iloc cannot enlarge its target object'
        with pytest.raises(IndexError, match=msg):
            s.iloc[3] = 5.0
        msg = 'index 3 is out of bounds for axis 0 with size 3'
        with pytest.raises(IndexError, match=msg):
            s.iat[3] = 5.0

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_partial_setting_frame(self, using_array_manager):
        df_orig = DataFrame(np.arange(6).reshape(3, 2), columns=['A', 'B'], dtype='int64')
        df = df_orig.copy()
        msg = 'iloc cannot enlarge its target object'
        with pytest.raises(IndexError, match=msg):
            df.iloc[4, 2] = 5.0
        msg = 'index 2 is out of bounds for axis 0 with size 2'
        if using_array_manager:
            msg = 'list index out of range'
        with pytest.raises(IndexError, match=msg):
            df.iat[4, 2] = 5.0
        expected = DataFrame({'A': [0, 4, 4], 'B': [1, 5, 5]})
        df = df_orig.copy()
        df.iloc[1] = df.iloc[2]
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'A': [0, 4, 4], 'B': [1, 5, 5]})
        df = df_orig.copy()
        df.loc[1] = df.loc[2]
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'A': [0, 2, 4, 4], 'B': [1, 3, 5, 5]})
        df = df_orig.copy()
        df.loc[3] = df.loc[2]
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'A': [0, 2, 4], 'B': [0, 2, 4]})
        df = df_orig.copy()
        df.loc[:, 'B'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'A': [0, 2, 4], 'B': Series([0.0, 2.0, 4.0])})
        df = df_orig.copy()
        df['B'] = df['B'].astype(np.float64)
        df.loc[:, 'B'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)
        expected = df_orig.copy()
        expected['C'] = df['A']
        df = df_orig.copy()
        df.loc[:, 'C'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)
        expected = df_orig.copy()
        expected['C'] = df['A']
        df = df_orig.copy()
        df.loc[:, 'C'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)

    def test_partial_setting2(self):
        dates = date_range('1/1/2000', periods=8)
        df_orig = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
        expected = pd.concat([df_orig, DataFrame({'A': 7}, index=dates[-1:] + dates.freq)], sort=True)
        df = df_orig.copy()
        df.loc[dates[-1] + dates.freq, 'A'] = 7
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.at[dates[-1] + dates.freq, 'A'] = 7
        tm.assert_frame_equal(df, expected)
        exp_other = DataFrame({0: 7}, index=dates[-1:] + dates.freq)
        expected = pd.concat([df_orig, exp_other], axis=1)
        df = df_orig.copy()
        df.loc[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.at[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)

    def test_partial_setting_mixed_dtype(self):
        df = DataFrame([[True, 1], [False, 2]], columns=['female', 'fitness'])
        s = df.loc[1].copy()
        s.name = 2
        expected = pd.concat([df, DataFrame(s).T.infer_objects()])
        df.loc[2] = df.loc[1]
        tm.assert_frame_equal(df, expected)

    def test_series_partial_set(self):
        ser = Series([0.1, 0.2], index=[1, 2])
        expected = Series([np.nan, 0.2, np.nan], index=[3, 2, 3])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[3, 2, 3]]
        result = ser.reindex([3, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected = Series([np.nan, 0.2, np.nan, np.nan], index=[3, 2, 3, 'x'])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[3, 2, 3, 'x']]
        result = ser.reindex([3, 2, 3, 'x'])
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected = Series([0.2, 0.2, 0.1], index=[2, 2, 1])
        result = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected = Series([0.2, 0.2, np.nan, 0.1], index=[2, 2, 'x', 1])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[2, 2, 'x', 1]]
        result = ser.reindex([2, 2, 'x', 1])
        tm.assert_series_equal(result, expected, check_index_type=True)
        msg = f'''\\"None of \\[Index\\(\\[3, 3, 3\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]
        expected = Series([0.2, 0.2, np.nan], index=[2, 2, 3])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[2, 2, 3]]
        result = ser.reindex([2, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s = Series([0.1, 0.2, 0.3], index=[1, 2, 3])
        expected = Series([0.3, np.nan, np.nan], index=[3, 4, 4])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[3, 4, 4]]
        result = s.reindex([3, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([np.nan, 0.3, 0.3], index=[5, 3, 3])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[5, 3, 3]]
        result = s.reindex([5, 3, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([np.nan, 0.4, 0.4], index=[5, 4, 4])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[5, 4, 4]]
        result = s.reindex([5, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s = Series([0.1, 0.2, 0.3, 0.4], index=[4, 5, 6, 7])
        expected = Series([0.4, np.nan, np.nan], index=[7, 2, 2])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[7, 2, 2]]
        result = s.reindex([7, 2, 2])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([0.4, np.nan, np.nan], index=[4, 5, 5])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[4, 5, 5]]
        result = s.reindex([4, 5, 5])
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected = Series([0.2, 0.2, 0.1, 0.1], index=[2, 2, 1, 1])
        result = ser.iloc[[1, 1, 0, 0]]
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_series_partial_set_with_name(self):
        idx = Index([1, 2], dtype='int64', name='idx')
        ser = Series([0.1, 0.2], index=idx, name='s')
        with pytest.raises(KeyError, match='\\[3\\] not in index'):
            ser.loc[[3, 2, 3]]
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[3, 2, 3, 'x']]
        exp_idx = Index([2, 2, 1], dtype='int64', name='idx')
        expected = Series([0.2, 0.2, 0.1], index=exp_idx, name='s')
        result = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)
        with pytest.raises(KeyError, match="\\['x'\\] not in index"):
            ser.loc[[2, 2, 'x', 1]]
        msg = f'''\\"None of \\[Index\\(\\[3, 3, 3\\], dtype='{np.dtype(int)}', name='idx'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[2, 2, 3]]
        idx = Index([1, 2, 3], dtype='int64', name='idx')
        with pytest.raises(KeyError, match='not in index'):
            Series([0.1, 0.2, 0.3], index=idx, name='s').loc[[3, 4, 4]]
        idx = Index([1, 2, 3, 4], dtype='int64', name='idx')
        with pytest.raises(KeyError, match='not in index'):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name='s').loc[[5, 3, 3]]
        idx = Index([1, 2, 3, 4], dtype='int64', name='idx')
        with pytest.raises(KeyError, match='not in index'):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name='s').loc[[5, 4, 4]]
        idx = Index([4, 5, 6, 7], dtype='int64', name='idx')
        with pytest.raises(KeyError, match='not in index'):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name='s').loc[[7, 2, 2]]
        idx = Index([1, 2, 3, 4], dtype='int64', name='idx')
        with pytest.raises(KeyError, match='not in index'):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name='s').loc[[4, 5, 5]]
        exp_idx = Index([2, 2, 1, 1], dtype='int64', name='idx')
        expected = Series([0.2, 0.2, 0.1, 0.1], index=exp_idx, name='s')
        result = ser.iloc[[1, 1, 0, 0]]
        tm.assert_series_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize('key', [100, 100.0])
    def test_setitem_with_expansion_numeric_into_datetimeindex(self, key):
        orig = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df = orig.copy()
        df.loc[key, :] = df.iloc[0]
        ex_index = Index(list(orig.index) + [key], dtype=object, name=orig.index.name)
        ex_data = np.concatenate([orig.values, df.iloc[[0]].values], axis=0)
        expected = DataFrame(ex_data, index=ex_index, columns=orig.columns)
        tm.assert_frame_equal(df, expected)

    def test_partial_set_invalid(self):
        orig = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df = orig.copy()
        df.loc['a', :] = df.iloc[0]
        ser = Series(df.iloc[0], name='a')
        exp = pd.concat([orig, DataFrame(ser).T.infer_objects()])
        tm.assert_frame_equal(df, exp)
        tm.assert_index_equal(df.index, Index(orig.index.tolist() + ['a']))
        assert df.index.dtype == 'object'

    @pytest.mark.parametrize('idx,labels,expected_idx', [(period_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-08', '2000-01-12'], [Period('2000-01-04', freq='D'), Period('2000-01-08', freq='D'), Period('2000-01-12', freq='D')]), (date_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-08', '2000-01-12'], [Timestamp('2000-01-04'), Timestamp('2000-01-08'), Timestamp('2000-01-12')]), (pd.timedelta_range(start='1 day', periods=20), ['4D', '8D', '12D'], [pd.Timedelta('4 day'), pd.Timedelta('8 day'), pd.Timedelta('12 day')])])
    def test_loc_with_list_of_strings_representing_datetimes(self, idx, labels, expected_idx, frame_or_series):
        obj = frame_or_series(range(20), index=idx)
        expected_value = [3, 7, 11]
        expected = frame_or_series(expected_value, expected_idx)
        tm.assert_equal(expected, obj.loc[labels])
        if frame_or_series is Series:
            tm.assert_series_equal(expected, obj[labels])

    @pytest.mark.parametrize('idx,labels', [(period_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-30']), (date_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-30']), (pd.timedelta_range(start='1 day', periods=20), ['3 day', '30 day'])])
    def test_loc_with_list_of_strings_representing_datetimes_missing_value(self, idx, labels):
        ser = Series(range(20), index=idx)
        df = DataFrame(range(20), index=idx)
        msg = 'not in index'
        with pytest.raises(KeyError, match=msg):
            ser.loc[labels]
        with pytest.raises(KeyError, match=msg):
            ser[labels]
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]

    @pytest.mark.parametrize('idx,labels,msg', [(period_range(start='2000', periods=20, freq='D'), Index(['4D', '8D'], dtype=object), "None of \\[Index\\(\\['4D', '8D'\\], dtype='object'\\)\\] are in the \\[index\\]"), (date_range(start='2000', periods=20, freq='D'), Index(['4D', '8D'], dtype=object), "None of \\[Index\\(\\['4D', '8D'\\], dtype='object'\\)\\] are in the \\[index\\]"), (pd.timedelta_range(start='1 day', periods=20), Index(['2000-01-04', '2000-01-08'], dtype=object), "None of \\[Index\\(\\['2000-01-04', '2000-01-08'\\], dtype='object'\\)\\] are in the \\[index\\]")])
    def test_loc_with_list_of_strings_representing_datetimes_not_matched_type(self, idx, labels, msg):
        ser = Series(range(20), index=idx)
        df = DataFrame(range(20), index=idx)
        with pytest.raises(KeyError, match=msg):
            ser.loc[labels]
        with pytest.raises(KeyError, match=msg):
            ser[labels]
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]