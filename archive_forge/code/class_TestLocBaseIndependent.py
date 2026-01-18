from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestLocBaseIndependent:

    def test_loc_npstr(self):
        df = DataFrame(index=date_range('2021', '2022'))
        result = df.loc[np.array(['2021/6/1'])[0]:]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('msg, key', [("Period\\('2019', 'Y-DEC'\\), 'foo', 'bar'", (Period(2019), 'foo', 'bar')), ("Period\\('2019', 'Y-DEC'\\), 'y1', 'bar'", (Period(2019), 'y1', 'bar')), ("Period\\('2019', 'Y-DEC'\\), 'foo', 'z1'", (Period(2019), 'foo', 'z1')), ("Period\\('2018', 'Y-DEC'\\), Period\\('2016', 'Y-DEC'\\), 'bar'", (Period(2018), Period(2016), 'bar')), ("Period\\('2018', 'Y-DEC'\\), 'foo', 'y1'", (Period(2018), 'foo', 'y1')), ("Period\\('2017', 'Y-DEC'\\), 'foo', Period\\('2015', 'Y-DEC'\\)", (Period(2017), 'foo', Period(2015))), ("Period\\('2017', 'Y-DEC'\\), 'z1', 'bar'", (Period(2017), 'z1', 'bar'))])
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg, key):
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df = DataFrame({'A': [Period(2019), 'x1', 'x2'], 'B': [Period(2018), Period(2016), 'y1'], 'C': [Period(2017), 'z1', Period(2015)], 'V1': [1, 2, 3], 'V2': [10, 20, 30]}).set_index(['A', 'B', 'C'])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self):
        df = DataFrame({'a': [1]})
        with pytest.raises(KeyError, match='א'):
            df.loc[:, 'א']

    def test_loc_getitem_dups(self):
        df = DataFrame(np.random.default_rng(2).random((20, 5)), index=['ABCDE'[x % 5] for x in range(20)])
        expected = df.loc['A', 0]
        result = df.loc[:, 0].loc['A']
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self):
        df = DataFrame([[1, 2, 'foo', 'bar', Timestamp('20130101')]], columns=['a', 'a', 'a', 'a', 'a'], index=[1])
        expected = Series([1, 2, 'foo', 'bar', Timestamp('20130101')], index=['a', 'a', 'a', 'a', 'a'], name=1)
        result = df.iloc[0]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self):
        df_orig = DataFrame({'me': list('rttti'), 'foo': list('aaade'), 'bar': np.arange(5, dtype='float64') * 1.34 + 2, 'bar2': np.arange(5, dtype='float64') * -0.34 + 2}).set_index('me')
        indexer = ('r', ['bar', 'bar2'])
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])
        indexer = ('r', 'bar')
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]
        indexer = ('t', ['bar', 'bar2'])
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self):
        df1 = DataFrame({'a': [0, 1, 1], 'b': Series([100, 200, 300], dtype='uint32')})
        ix = df1['a'] == 1
        newb1 = df1.loc[ix, 'b'] + 1
        df1.loc[ix, 'b'] = newb1
        expected = DataFrame({'a': [0, 1, 1], 'b': Series([100, 201, 301], dtype='uint32')})
        tm.assert_frame_equal(df1, expected)
        df2 = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
        ix = df1['a'] == 1
        newb2 = df2.loc[ix, 'b']
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            df1.loc[ix, 'b'] = newb2
        expected = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
        tm.assert_frame_equal(df2, expected)

    def test_loc_setitem_dtype(self):
        df = DataFrame({'id': ['A'], 'a': [1.2], 'b': [0.0], 'c': [-2.5]})
        cols = ['a', 'b', 'c']
        df.loc[:, cols] = df.loc[:, cols].astype('float32')
        expected = DataFrame({'id': ['A'], 'a': np.array([1.2], dtype='float64'), 'b': np.array([0.0], dtype='float64'), 'c': np.array([-2.5], dtype='float64')})
        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self):
        s = Series(range(3), index=['a', 'b', 'c'])
        with pytest.raises(KeyError, match='not in index'):
            s[['a', 'd']]
        s = Series(range(3))
        with pytest.raises(KeyError, match='not in index'):
            s[[0, 3]]

    @pytest.mark.parametrize('index', [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index):
        s = Series([1, 2, 3])
        msg = f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self):
        pass

    def test_loc_to_fail(self):
        df = DataFrame(np.random.default_rng(2).random((3, 3)), index=['a', 'b', 'c'], columns=['e', 'f', 'g'])
        msg = f'''\\"None of \\[Index\\(\\[1, 2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self):
        s = Series(dtype=object)
        s.loc[1] = 1
        s.loc['a'] = 2
        with pytest.raises(KeyError, match='^-1$'):
            s.loc[-1]
        msg = f'''\\"None of \\[Index\\(\\[-1, -2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]
        msg = '\\"None of \\[Index\\(\\[\'4\'\\], dtype=\'object\'\\)\\] are in the \\[index\\]\\"'
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(['4'], dtype=object)]
        s.loc[-1] = 3
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[-1, -2]]
        s['a'] = 2
        msg = f'''\\"None of \\[Index\\(\\[-2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]
        del s['a']
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self):
        df = DataFrame([['a'], ['b']], index=[1, 2], columns=['value'])
        msg = f'''\\"None of \\[Index\\(\\[3\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]
        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self):
        s = Series([1, 2, 3])
        s.loc[[2]]
        msg = f""""None of [Index([3], dtype='{np.dtype(int)}')] are in the [index]"""
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[2, 3]]

    def test_loc_index(self):
        df = DataFrame(np.random.default_rng(2).random(size=(5, 10)), index=['alpha_0', 'alpha_1', 'alpha_2', 'beta_0', 'beta_1'])
        mask = df.index.map(lambda x: 'alpha' in x)
        expected = df.loc[np.array(mask)]
        result = df.loc[mask]
        tm.assert_frame_equal(result, expected)
        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)
        result = df.loc[pd.array(mask, dtype='boolean')]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self):
        df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=['A', 'B', 'C', 'D'], index=['A', 'B', 'C', 'D'])
        result = df.loc[:, 'A':'B'].iloc[0:2, :]
        assert (result.columns == ['A', 'B']).all()
        assert (result.index == ['A', 'B']).all()
        result = DataFrame({'a': [Timestamp('20130101')], 'b': [1]}).iloc[0]
        expected = Series([Timestamp('20130101'), 1], index=['a', 'b'], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self):
        return DataFrame({'date': date_range('2000-01-01', '2000-01-5'), 'val': Series(range(5), dtype=np.int64)})

    @pytest.mark.parametrize('val', [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)])
    def test_loc_setitem_consistency(self, frame_for_consistency, val):
        expected = DataFrame({'date': Series(0, index=range(5), dtype=np.int64), 'val': Series(range(5), dtype=np.int64)})
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df.loc[:, 'date'] = val
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency):
        expected = DataFrame({'date': Series('foo', index=range(5)), 'val': Series(range(5), dtype=np.int64)})
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df.loc[:, 'date'] = 'foo'
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency):
        expected = DataFrame({'date': Series(1.0, index=range(5)), 'val': Series(range(5), dtype=np.int64)})
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df.loc[:, 'date'] = 1.0
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_single_row(self):
        df = DataFrame({'date': Series([Timestamp('20180101')])})
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df.loc[:, 'date'] = 'string'
        expected = DataFrame({'date': Series(['string'])})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_empty(self):
        expected = DataFrame(columns=['x', 'y'])
        df = DataFrame(columns=['x', 'y'])
        with tm.assert_produces_warning(None):
            df.loc[:, 'x'] = 1
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=['x', 'y'])
        df['x'] = 1
        expected['x'] = expected['x'].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_slice_column_len(self):
        levels = [['Region_1'] * 4, ['Site_1', 'Site_1', 'Site_2', 'Site_2'], [3987227376, 3980680971, 3977723249, 3977723089]]
        mi = MultiIndex.from_arrays(levels, names=['Region', 'Site', 'RespondentID'])
        clevels = [['Respondent', 'Respondent', 'Respondent', 'OtherCat', 'OtherCat'], ['Something', 'StartDate', 'EndDate', 'Yes/No', 'SomethingElse']]
        cols = MultiIndex.from_arrays(clevels, names=['Level_0', 'Level_1'])
        values = [['A', '5/25/2015 10:59', '5/25/2015 11:22', 'Yes', np.nan], ['A', '5/21/2015 9:40', '5/21/2015 9:52', 'Yes', 'Yes'], ['A', '5/20/2015 8:27', '5/20/2015 8:41', 'Yes', np.nan], ['A', '5/20/2015 8:33', '5/20/2015 9:09', 'Yes', 'No']]
        df = DataFrame(values, index=mi, columns=cols)
        df.loc[:, ('Respondent', 'StartDate')] = to_datetime(df.loc[:, ('Respondent', 'StartDate')])
        df.loc[:, ('Respondent', 'EndDate')] = to_datetime(df.loc[:, ('Respondent', 'EndDate')])
        df = df.infer_objects(copy=False)
        df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent', 'EndDate')] - df.loc[:, ('Respondent', 'StartDate')]
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent', 'Duration')] / Timedelta(60000000000)
        expected = Series([23.0, 12.0, 14.0, 36.0], index=df.index, name=('Respondent', 'Duration'))
        tm.assert_series_equal(df['Respondent', 'Duration'], expected)

    @pytest.mark.parametrize('unit', ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us'])
    def test_loc_assign_non_ns_datetime(self, unit):
        df = DataFrame({'timestamp': [np.datetime64('2017-02-11 12:41:29'), np.datetime64('1991-11-07 04:22:37')]})
        df.loc[:, unit] = df.loc[:, 'timestamp'].values.astype(f'datetime64[{unit}]')
        df['expected'] = df.loc[:, 'timestamp'].values.astype(f'datetime64[{unit}]')
        expected = Series(df.loc[:, 'expected'], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self):
        df = DataFrame.from_dict({'date': [1485264372711, 1485265925110, 1540215845888, 1540282121025]})
        df['date_dt'] = to_datetime(df['date'], unit='ms', cache=True)
        df.loc[:, 'date_dt_cp'] = df.loc[:, 'date_dt']
        df.loc[[2, 3], 'date_dt_cp'] = df.loc[[2, 3], 'date_dt']
        expected = DataFrame([[1485264372711, '2017-01-24 13:26:12.711', '2017-01-24 13:26:12.711'], [1485265925110, '2017-01-24 13:52:05.110', '2017-01-24 13:52:05.110'], [1540215845888, '2018-10-22 13:44:05.888', '2018-10-22 13:44:05.888'], [1540282121025, '2018-10-23 08:08:41.025', '2018-10-23 08:08:41.025']], columns=['date', 'date_dt', 'date_dt_cp'])
        columns = ['date_dt', 'date_dt_cp']
        expected[columns] = expected[columns].apply(to_datetime)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self):
        df = DataFrame(index=[3, 5, 4], columns=['A'], dtype=float)
        df.loc[[4, 3, 5], 'A'] = np.array([1, 2, 3], dtype='int64')
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({'A': ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self):
        df = DataFrame(index=[3, 5, 4], columns=['A', 'B'], dtype=float)
        df['B'] = 'string'
        df.loc[[4, 3, 5], 'A'] = np.array([1, 2, 3], dtype='int64')
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype='int64')
        expected = DataFrame({'A': ser.astype(float)})
        expected['B'] = 'string'
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self):
        df = DataFrame(index=[1, 2, 3], columns=['A', 'B'], dtype=float)
        df['B'] = 'string'
        df.loc[slice(3, 0, -1), 'A'] = np.array([1, 2, 3], dtype='int64')
        expected = DataFrame({'A': [3.0, 2.0, 1.0], 'B': 'string'}, index=[1, 2, 3])
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_frame(self):
        keys1 = ['@' + str(i) for i in range(5)]
        val1 = np.arange(5, dtype='int64')
        keys2 = ['@' + str(i) for i in range(4)]
        val2 = np.arange(4, dtype='int64')
        index = list(set(keys1).union(keys2))
        df = DataFrame(index=index)
        df['A'] = np.nan
        df.loc[keys1, 'A'] = val1
        df['B'] = np.nan
        df.loc[keys2, 'B'] = val2
        sera = Series(val1, index=keys1, dtype=np.float64)
        serb = Series(val2, index=keys2)
        expected = DataFrame({'A': sera, 'B': serb}, columns=Index(['A', 'B'], dtype=object)).reindex(index=index)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=list('abcd'), columns=list('ABCD'))
        result = df.iloc[0, 0]
        df.loc['a', 'A'] = 1
        result = df.loc['a', 'A']
        assert result == 1
        result = df.iloc[0, 0]
        assert result == 1
        df.loc[:, 'B':'D'] = 0
        expected = df.loc[:, 'B':'D']
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self):
        df = DataFrame({'A': [1, 2, 3], 'B': np.nan})
        df.loc[df.B > df.A, 'B'] = df.A
        expected = DataFrame({'A': [1, 2, 3], 'B': np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self):
        df = DataFrame({1: [1, 2], 2: [3, 4], 'a': ['a', 'b']})
        result = df.loc[0, [1, 2]]
        expected = Series([1, 3], index=Index([1, 2], dtype=object), dtype=object, name=0)
        tm.assert_series_equal(result, expected)
        expected = DataFrame({1: [5, 2], 2: [6, 4], 'a': ['a', 'b']})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_loc_setitem_frame_multiples(self, warn_copy_on_write):
        df = DataFrame({'A': ['foo', 'bar', 'baz'], 'B': Series(range(3), dtype=np.int64)})
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame({'A': ['bar', 'baz', 'baz'], 'B': Series([1, 2, 2], dtype=np.int64)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'date': date_range('2000-01-01', '2000-01-5'), 'val': Series(range(5), dtype=np.int64)})
        expected = DataFrame({'date': [Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000103')], 'val': Series([0, 1, 0, 1, 2], dtype=np.int64)})
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [['A'], slice(None, 'A', None), np.array(['A'])])
    @pytest.mark.parametrize('value', [['Z'], np.array(['Z'])])
    def test_loc_setitem_with_scalar_index(self, indexer, value):
        df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B']).astype({'A': object})
        df.loc[0, indexer] = value
        result = df.loc[0, 'A']
        assert is_scalar(result) and result == 'Z'

    @pytest.mark.parametrize('index,box,expected', [(([0, 2], ['A', 'B', 'C', 'D']), 7, DataFrame([[7, 7, 7, 7], [3, 4, np.nan, np.nan], [7, 7, 7, 7]], columns=['A', 'B', 'C', 'D'])), ((1, ['C', 'D']), [7, 8], DataFrame([[1, 2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]], columns=['A', 'B', 'C', 'D'])), ((1, ['A', 'B', 'C']), np.array([7, 8, 9], dtype=np.int64), DataFrame([[1, 2, np.nan], [7, 8, 9], [5, 6, np.nan]], columns=['A', 'B', 'C'])), ((slice(1, 3, None), ['B', 'C', 'D']), [[7, 8, 9], [10, 11, 12]], DataFrame([[1, 2, np.nan, np.nan], [3, 7, 8, 9], [5, 10, 11, 12]], columns=['A', 'B', 'C', 'D'])), ((slice(1, 3, None), ['C', 'A', 'D']), np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64), DataFrame([[1, 2, np.nan, np.nan], [8, 4, 7, 9], [11, 6, 10, 12]], columns=['A', 'B', 'C', 'D'])), ((slice(None, None, None), ['A', 'C']), DataFrame([[7, 8], [9, 10], [11, 12]], columns=['A', 'C']), DataFrame([[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=['A', 'B', 'C']))])
    def test_loc_setitem_missing_columns(self, index, box, expected):
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A', 'B'])
        df.loc[index] = box
        tm.assert_frame_equal(df, expected)

    def test_loc_coercion(self):
        df = DataFrame({'date': [Timestamp('20130101').tz_localize('UTC'), pd.NaT]})
        expected = df.dtypes
        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion2(self):
        df = DataFrame({'date': [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        expected = df.dtypes
        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion3(self):
        df = DataFrame({'text': ['some words'] + [None] * 9})
        expected = df.dtypes
        result = df.iloc[0:2]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[3:]
        tm.assert_series_equal(result.dtypes, expected)

    def test_setitem_new_key_tz(self, indexer_sl):
        vals = [to_datetime(42).tz_localize('UTC'), to_datetime(666).tz_localize('UTC')]
        expected = Series(vals, index=Index(['foo', 'bar'], dtype=object))
        ser = Series(dtype=object)
        indexer_sl(ser)['foo'] = vals[0]
        indexer_sl(ser)['bar'] = vals[1]
        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self):
        df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3])
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:]
        msg = "'Cannot get left slice bound for non-unique label: 0'"
        with pytest.raises(KeyError, match=msg):
            df.loc[0:]
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:2]
        df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]).sort_index(axis=0)
        result = df.loc[1:]
        expected = DataFrame({'A': [2, 4, 5, 6], 'B': [4, 6, 7, 8]}, index=[1, 1, 2, 3])
        tm.assert_frame_equal(result, expected)
        result = df.loc[0:]
        tm.assert_frame_equal(result, df)
        result = df.loc[1:2]
        expected = DataFrame({'A': [2, 4, 5], 'B': [4, 6, 7]}, index=[1, 1, 2])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize('length, l2', [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(self, length, l2):
        columns = list('ABCDEFG')
        df = pd.concat([DataFrame(np.random.default_rng(2).standard_normal((length, len(columns))), index=np.arange(length), columns=columns), DataFrame(np.ones((l2, len(columns))), index=[0] * l2, columns=columns)])
        assert df.index.is_unique is False
        mask = np.arange(l2)
        result = df.loc[mask]
        expected = pd.concat([df.take([0]), DataFrame(np.ones((len(mask), len(columns))), index=[0] * len(mask), columns=columns), df.take(mask[1:])])
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self):
        df = DataFrame([[1, 1], [1, 1]])
        df.index.name = 'index_name'
        result = df.iloc[[0, 1]].index.name
        assert result == 'index_name'
        result = df.loc[[0, 1]].index.name
        assert result == 'index_name'

    def test_loc_empty_list_indexer_is_ok(self):
        df = DataFrame(np.ones((5, 2)), index=Index([f'i-{i}' for i in range(5)], name='a'), columns=Index([f'i-{i}' for i in range(2)], name='a'))
        tm.assert_frame_equal(df.loc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.loc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.loc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True)

    def test_identity_slice_returns_new_object(self, using_copy_on_write, warn_copy_on_write):
        original_df = DataFrame({'a': [1, 2, 3]})
        sliced_df = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df
        assert np.shares_memory(original_df['a']._values, sliced_df['a']._values)
        with tm.assert_cow_warning(warn_copy_on_write):
            original_df.loc[:, 'a'] = [4, 4, 4]
        if using_copy_on_write:
            assert (sliced_df['a'] == [1, 2, 3]).all()
        else:
            assert (sliced_df['a'] == 4).all()
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        if using_copy_on_write or warn_copy_on_write:
            assert df[0] is not df.loc[:, 0]
        else:
            assert df[0] is df.loc[:, 0]
        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series
        with tm.assert_cow_warning(warn_copy_on_write):
            original_series[:3] = [7, 8, 9]
        if using_copy_on_write:
            assert all(sliced_series[:3] == [1, 2, 3])
        else:
            assert all(sliced_series[:3] == [7, 8, 9])

    def test_loc_copy_vs_view(self, request, using_copy_on_write):
        if not using_copy_on_write:
            mark = pytest.mark.xfail(reason='accidental fix reverted - GH37497')
            request.applymarker(mark)
        x = DataFrame(zip(range(3), range(3)), columns=['a', 'b'])
        y = x.copy()
        q = y.loc[:, 'a']
        q += 2
        tm.assert_frame_equal(x, y)
        z = x.copy()
        q = z.loc[x.index, 'a']
        q += 2
        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self):
        umax = np.iinfo('uint64').max
        ser = Series([1, 2], index=[umax - 1, umax])
        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        assert result == expected
        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)
        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self):
        umax = np.iinfo('uint64').max
        ser = Series([1, 2], index=[umax - 1, umax])
        with pytest.raises(KeyError, match='-1'):
            ser.loc[-1]
        with pytest.raises(KeyError, match='-1'):
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self):
        data = [1, 2, 3]
        expected = DataFrame({'x': data, 'y': np.array([np.nan] * len(data), dtype=object)})
        df = DataFrame(columns=['x', 'y'])
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self):
        data = [1, 2, 3]
        expected = DataFrame({'x': data, 'y': np.array([np.nan] * len(data), dtype=object)})
        df = DataFrame(columns=['x', 'y'])
        df['x'] = df['x'].astype(np.int64)
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self):
        expected = DataFrame({'x': [1.0], 'y': [np.nan]})
        df = DataFrame(columns=['x', 'y'], dtype=float)
        df.loc[0, 'x'] = expected.loc[0, 'x']
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self):
        data = [1, 2]
        df = DataFrame(columns=['x', 'y'])
        df.index = df.index.astype(np.int64)
        msg = f"None of \\[Index\\(\\[0, 1\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]"
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], 'x'] = data
        msg = 'setting an array element with a sequence.'
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, 'x'] = data

    def test_indexing_zerodim_np_array(self):
        df = DataFrame([[1, 2], [3, 4]])
        result = df.loc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        s = Series([1, 2])
        result = s.loc[np.array(0)]
        assert result == 1

    def test_loc_reverse_assignment(self):
        data = [1, 2, 3, 4, 5, 6] + [None] * 4
        expected = Series(data, index=range(2010, 2020))
        result = Series(index=range(2010, 2020), dtype=np.float64)
        result.loc[2015:2010:-1] = [6, 5, 4, 3, 2, 1]
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set int into string")
    def test_loc_setitem_str_to_small_float_conversion_type(self):
        col_data = [str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)]
        result = DataFrame(col_data, columns=['A'])
        expected = DataFrame(col_data, columns=['A'], dtype=object)
        tm.assert_frame_equal(result, expected)
        result.loc[result.index, 'A'] = [float(x) for x in col_data]
        expected = DataFrame(col_data, columns=['A'], dtype=float).astype(object)
        tm.assert_frame_equal(result, expected)
        result['A'] = [float(x) for x in col_data]
        expected = DataFrame(col_data, columns=['A'], dtype=float)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_time_object(self, frame_or_series):
        rng = date_range('1/1/2000', '1/5/2000', freq='5min')
        mask = (rng.hour == 9) & (rng.minute == 30)
        obj = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng)
        obj = tm.get_obj(obj, frame_or_series)
        result = obj.loc[time(9, 30)]
        exp = obj.loc[mask]
        tm.assert_equal(result, exp)
        chunk = obj.loc['1/4/2000':]
        result = chunk.loc[time(9, 30)]
        expected = result[-1:]
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('spmatrix_t', ['coo_matrix', 'csc_matrix', 'csr_matrix'])
    @pytest.mark.parametrize('dtype', [np.int64, np.float64, complex])
    def test_loc_getitem_range_from_spmatrix(self, spmatrix_t, dtype):
        sp_sparse = pytest.importorskip('scipy.sparse')
        spmatrix_t = getattr(sp_sparse, spmatrix_t)
        rows, cols = (5, 7)
        spmatrix = spmatrix_t(np.eye(rows, cols, dtype=dtype), dtype=dtype)
        df = DataFrame.sparse.from_spmatrix(spmatrix)
        itr_idx = range(2, rows)
        result = df.loc[itr_idx].values
        expected = spmatrix.toarray()[itr_idx]
        tm.assert_numpy_array_equal(result, expected)
        result = df.loc[itr_idx].dtypes.values
        expected = np.full(cols, SparseDtype(dtype, fill_value=0))
        tm.assert_numpy_array_equal(result, expected)

    def test_loc_getitem_listlike_all_retains_sparse(self):
        df = DataFrame({'A': pd.array([0, 0], dtype=SparseDtype('int64'))})
        result = df.loc[[0, 1]]
        tm.assert_frame_equal(result, df)

    def test_loc_getitem_sparse_frame(self):
        sp_sparse = pytest.importorskip('scipy.sparse')
        df = DataFrame.sparse.from_spmatrix(sp_sparse.eye(5))
        result = df.loc[range(2)]
        expected = DataFrame([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype=SparseDtype('float64', 0.0))
        tm.assert_frame_equal(result, expected)
        result = df.loc[range(2)].loc[range(1)]
        expected = DataFrame([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=SparseDtype('float64', 0.0))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sparse_series(self):
        s = Series([1.0, 0.0, 0.0, 0.0, 0.0], dtype=SparseDtype('float64', 0.0))
        result = s.loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype('float64', 0.0))
        tm.assert_series_equal(result, expected)
        result = s.loc[range(3)].loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype('float64', 0.0))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('indexer', ['loc', 'iloc'])
    def test_getitem_single_row_sparse_df(self, indexer):
        df = DataFrame([[1.0, 0.0, 1.5], [0.0, 2.0, 0.0]], dtype=SparseDtype(float))
        result = getattr(df, indexer)[0]
        expected = Series([1.0, 0.0, 1.5], dtype=SparseDtype(float), name=0)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('key_type', [iter, np.array, Series, Index])
    def test_loc_getitem_iterable(self, float_frame, key_type):
        idx = key_type(['A', 'B', 'C'])
        result = float_frame.loc[:, idx]
        expected = float_frame.loc[:, ['A', 'B', 'C']]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_timedelta_0seconds(self):
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 4)))
        df.index = timedelta_range(start='0s', periods=10, freq='s')
        expected = df.loc[Timedelta('0s'):, :]
        result = df.loc['0s':, :]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('val,expected', [(2 ** 63 - 1, Series([1])), (2 ** 63, Series([2]))])
    def test_loc_getitem_uint64_scalar(self, val, expected):
        df = DataFrame([1, 2], index=[2 ** 63 - 1, 2 ** 63])
        result = df.loc[val]
        expected.name = val
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_int_label_with_float_index(self, float_numpy_dtype):
        dtype = float_numpy_dtype
        ser = Series(['a', 'b', 'c'], index=Index([0, 0.5, 1], dtype=dtype))
        expected = ser.copy()
        ser.loc[1] = 'zoo'
        expected.iloc[2] = 'zoo'
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('indexer, expected', [(0, [20, 1, 2, 3, 4, 5, 6, 7, 8, 9]), (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]), ([3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9])])
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer, expected):
        tdi = to_timedelta(range(10), unit='s')
        df = DataFrame({'x': range(10)}, dtype='int64', index=tdi)
        df.loc[df.index[indexer], 'x'] = 20
        expected = DataFrame(expected, index=tdi, columns=['x'], dtype='int64')
        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_categorical_values_partial_column_slice(self):
        df = DataFrame({'a': [1, 1, 1, 1, 1], 'b': list('aaaaa')})
        exp = DataFrame({'a': [1, 'b', 'b', 1, 1], 'b': list('aabba')})
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            df.loc[1:2, 'a'] = Categorical(['b', 'b'], categories=['a', 'b'])
            df.loc[2:3, 'b'] = Categorical(['b', 'b'], categories=['a', 'b'])
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_single_row_categorical(self, using_infer_string):
        df = DataFrame({'Alpha': ['a'], 'Numeric': [0]})
        categories = Categorical(df['Alpha'], categories=['a', 'b', 'c'])
        df.loc[:, 'Alpha'] = categories
        result = df['Alpha']
        expected = Series(categories, index=df.index, name='Alpha').astype(object if not using_infer_string else 'string[pyarrow_numpy]')
        tm.assert_series_equal(result, expected)
        df['Alpha'] = categories
        tm.assert_series_equal(df['Alpha'], Series(categories, name='Alpha'))

    def test_loc_setitem_datetime_coercion(self):
        df = DataFrame({'c': [Timestamp('2010-10-01')] * 3})
        df.loc[0:1, 'c'] = np.datetime64('2008-08-08')
        assert Timestamp('2008-08-08') == df.loc[0, 'c']
        assert Timestamp('2008-08-08') == df.loc[1, 'c']
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df.loc[2, 'c'] = date(2005, 5, 5)
        assert Timestamp('2005-05-05').date() == df.loc[2, 'c']

    @pytest.mark.parametrize('idxer', ['var', ['var']])
    def test_loc_setitem_datetimeindex_tz(self, idxer, tz_naive_fixture):
        tz = tz_naive_fixture
        idx = date_range(start='2015-07-12', periods=3, freq='h', tz=tz)
        expected = DataFrame(1.2, index=idx, columns=['var'])
        result = DataFrame(index=idx, columns=['var'], dtype=np.float64)
        with tm.assert_produces_warning(FutureWarning if idxer == 'var' else None, match='incompatible dtype'):
            result.loc[:, idxer] = expected
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_time_key(self, using_array_manager):
        index = date_range('2012-01-01', '2012-01-05', freq='30min')
        df = DataFrame(np.random.default_rng(2).standard_normal((len(index), 5)), index=index)
        akey = time(12, 0, 0)
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        ainds = [24, 72, 120, 168]
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]
        result = df.copy()
        result.loc[akey] = 0
        result = result.loc[akey]
        expected = df.loc[akey].copy()
        expected.loc[:] = 0
        if using_array_manager:
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.loc[akey] = 0
        result.loc[akey] = df.iloc[ainds]
        tm.assert_frame_equal(result, df)
        result = df.copy()
        result.loc[bkey] = 0
        result = result.loc[bkey]
        expected = df.loc[bkey].copy()
        expected.loc[:] = 0
        if using_array_manager:
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.loc[bkey] = 0
        result.loc[bkey] = df.iloc[binds]
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize('key', ['A', ['A'], ('A', slice(None))])
    def test_loc_setitem_unsorted_multiindex_columns(self, key):
        mi = MultiIndex.from_tuples([('A', 4), ('B', '3'), ('A', '2')])
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        obj = df.copy()
        obj.loc[:, key] = np.zeros((2, 2), dtype='int64')
        expected = DataFrame([[0, 2, 0], [0, 5, 0]], columns=mi)
        tm.assert_frame_equal(obj, expected)
        df = df.sort_index(axis=1)
        df.loc[:, key] = np.zeros((2, 2), dtype='int64')
        expected = expected.sort_index(axis=1)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_uint_drop(self, any_int_numpy_dtype):
        series = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        series.loc[0] = 4
        expected = Series([4, 2, 3], dtype=any_int_numpy_dtype)
        tm.assert_series_equal(series, expected)

    def test_loc_setitem_td64_non_nano(self):
        ser = Series(10 * [np.timedelta64(10, 'm')])
        ser.loc[[1, 2, 3]] = np.timedelta64(20, 'm')
        expected = Series(10 * [np.timedelta64(10, 'm')])
        expected.loc[[1, 2, 3]] = Timedelta(np.timedelta64(20, 'm'))
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_2d_to_1d_raises(self):
        data = np.random.default_rng(2).standard_normal((2, 2))
        ser = Series(range(2), dtype='float64')
        msg = 'setting an array element with a sequence.'
        with pytest.raises(ValueError, match=msg):
            ser.loc[range(2)] = data
        with pytest.raises(ValueError, match=msg):
            ser.loc[:] = data

    def test_loc_getitem_interval_index(self):
        index = pd.interval_range(start=0, periods=3)
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=['A', 'B', 'C'])
        expected = 1
        result = df.loc[0.5, 'A']
        tm.assert_almost_equal(result, expected)

    def test_loc_getitem_interval_index2(self):
        index = pd.interval_range(start=0, periods=3, closed='both')
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=['A', 'B', 'C'])
        index_exp = pd.interval_range(start=0, periods=2, freq=1, closed='both')
        expected = Series([1, 4], index=index_exp, name='A')
        result = df.loc[1, 'A']
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tpl', [(1,), (1, 2)])
    def test_loc_getitem_index_single_double_tuples(self, tpl):
        idx = Index([(1,), (1, 2)], name='A', tupleize_cols=False)
        df = DataFrame(index=idx)
        result = df.loc[[tpl]]
        idx = Index([tpl], name='A', tupleize_cols=False)
        expected = DataFrame(index=idx)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_index_namedtuple(self):
        IndexType = namedtuple('IndexType', ['a', 'b'])
        idx1 = IndexType('foo', 'bar')
        idx2 = IndexType('baz', 'bof')
        index = Index([idx1, idx2], name='composite_index', tupleize_cols=False)
        df = DataFrame([(1, 2), (3, 4)], index=index, columns=['A', 'B'])
        result = df.loc[IndexType('foo', 'bar')]['A']
        assert result == 1

    def test_loc_setitem_single_column_mixed(self, using_infer_string):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['a', 'b', 'c', 'd', 'e'], columns=['foo', 'bar', 'baz'])
        df['str'] = 'qux'
        df.loc[df.index[::2], 'str'] = np.nan
        expected = Series([np.nan, 'qux', np.nan, 'qux', np.nan], dtype=object if not using_infer_string else 'string[pyarrow_numpy]').values
        tm.assert_almost_equal(df['str'].values, expected)

    def test_loc_setitem_cast2(self):
        df = DataFrame(np.random.default_rng(2).random((30, 3)), columns=tuple('ABC'))
        df['event'] = np.nan
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            df.loc[10, 'event'] = 'foo'
        result = df.dtypes
        expected = Series([np.dtype('float64')] * 3 + [np.dtype('object')], index=['A', 'B', 'C', 'event'])
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_cast3(self):
        df = DataFrame({'one': np.arange(6, dtype=np.int8)})
        df.loc[1, 'one'] = 6
        assert df.dtypes.one == np.dtype(np.int8)
        df.one = np.int8(7)
        assert df.dtypes.one == np.dtype(np.int8)

    def test_loc_setitem_range_key(self, frame_or_series):
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])
        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]
        obj.loc[range(3)] = values
        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_numpy_frame_categorical_value(self):
        df = DataFrame({'a': [1, 1, 1, 1, 1], 'b': ['a', 'a', 'a', 'a', 'a']})
        df.loc[1:2, 'a'] = Categorical([2, 2], categories=[1, 2])
        expected = DataFrame({'a': [1, 2, 2, 1, 1], 'b': ['a', 'a', 'a', 'a', 'a']})
        tm.assert_frame_equal(df, expected)