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
class TestDataFrameSetItemWithExpansion:

    def test_setitem_listlike_views(self, using_copy_on_write, warn_copy_on_write):
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 4, 6]})
        ser = df['a']
        df[['c', 'd']] = np.array([[0.1, 0.2], [0.3, 0.4], [0.4, 0.5]])
        with tm.assert_cow_warning(warn_copy_on_write):
            df.iloc[0, 0] = 100
        if using_copy_on_write:
            expected = Series([1, 2, 3], name='a')
        else:
            expected = Series([100, 2, 3], name='a')
        tm.assert_series_equal(ser, expected)

    def test_setitem_string_column_numpy_dtype_raising(self):
        df = DataFrame([[1, 2], [3, 4]])
        df['0 - Name'] = [5, 6]
        expected = DataFrame([[1, 2, 5], [3, 4, 6]], columns=[0, 1, '0 - Name'])
        tm.assert_frame_equal(df, expected)

    def test_setitem_empty_df_duplicate_columns(self, using_copy_on_write):
        df = DataFrame(columns=['a', 'b', 'b'], dtype='float64')
        df.loc[:, 'a'] = list(range(2))
        expected = DataFrame([[0, np.nan, np.nan], [1, np.nan, np.nan]], columns=['a', 'b', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_expansion_categorical_dtype(self):
        df = DataFrame({'value': np.array(np.random.default_rng(2).integers(0, 10000, 100), dtype='int32')})
        labels = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
        df = df.sort_values(by=['value'], ascending=True)
        ser = cut(df.value, range(0, 10500, 500), right=False, labels=labels)
        cat = ser.values
        df['D'] = cat
        result = df.dtypes
        expected = Series([np.dtype('int32'), CategoricalDtype(categories=labels, ordered=False)], index=['value', 'D'])
        tm.assert_series_equal(result, expected)
        df['E'] = ser
        result = df.dtypes
        expected = Series([np.dtype('int32'), CategoricalDtype(categories=labels, ordered=False), CategoricalDtype(categories=labels, ordered=False)], index=['value', 'D', 'E'])
        tm.assert_series_equal(result, expected)
        result1 = df['D']
        result2 = df['E']
        tm.assert_categorical_equal(result1._mgr.array, cat)
        ser.name = 'E'
        tm.assert_series_equal(result2.sort_index(), ser.sort_index())

    def test_setitem_scalars_no_index(self):
        df = DataFrame()
        df['foo'] = 1
        expected = DataFrame(columns=['foo']).astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_setitem_newcol_tuple_key(self, float_frame):
        assert ('A', 'B') not in float_frame.columns
        float_frame['A', 'B'] = float_frame['A']
        assert ('A', 'B') in float_frame.columns
        result = float_frame['A', 'B']
        expected = float_frame['A']
        tm.assert_series_equal(result, expected, check_names=False)

    def test_frame_setitem_newcol_timestamp(self):
        columns = date_range(start='1/1/2012', end='2/1/2012', freq=BDay())
        data = DataFrame(columns=columns, index=range(10))
        t = datetime(2012, 11, 1)
        ts = Timestamp(t)
        data[ts] = np.nan
        assert np.isnan(data[ts]).all()

    def test_frame_setitem_rangeindex_into_new_col(self):
        df = DataFrame({'a': ['a', 'b']})
        df['b'] = df.index
        df.loc[[False, True], 'b'] = 100
        result = df.loc[[1], :]
        expected = DataFrame({'a': ['b'], 'b': [100]}, index=[1])
        tm.assert_frame_equal(result, expected)

    def test_setitem_frame_keep_ea_dtype(self, any_numeric_ea_dtype):
        df = DataFrame(columns=['a', 'b'], data=[[1, 2], [3, 4]])
        df['c'] = DataFrame({'a': [10, 11]}, dtype=any_numeric_ea_dtype)
        expected = DataFrame({'a': [1, 3], 'b': [2, 4], 'c': Series([10, 11], dtype=any_numeric_ea_dtype)})
        tm.assert_frame_equal(df, expected)

    def test_loc_expansion_with_timedelta_type(self):
        result = DataFrame(columns=list('abc'))
        result.loc[0] = {'a': pd.to_timedelta(5, unit='s'), 'b': pd.to_timedelta(72, unit='s'), 'c': '23'}
        expected = DataFrame([[pd.Timedelta('0 days 00:00:05'), pd.Timedelta('0 days 00:01:12'), '23']], index=Index([0]), columns=['a', 'b', 'c'])
        tm.assert_frame_equal(result, expected)