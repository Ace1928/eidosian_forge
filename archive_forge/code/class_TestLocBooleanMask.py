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
class TestLocBooleanMask:

    def test_loc_setitem_bool_mask_timedeltaindex(self):
        df = DataFrame({'x': range(10)})
        df.index = to_timedelta(range(10), unit='s')
        conditions = [df['x'] > 3, df['x'] == 3, df['x'] < 3]
        expected_data = [[0, 1, 2, 3, 10, 10, 10, 10, 10, 10], [0, 1, 2, 10, 4, 5, 6, 7, 8, 9], [10, 10, 10, 3, 4, 5, 6, 7, 8, 9]]
        for cond, data in zip(conditions, expected_data):
            result = df.copy()
            result.loc[cond, 'x'] = 10
            expected = DataFrame(data, index=to_timedelta(range(10), unit='s'), columns=['x'], dtype='int64')
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_loc_setitem_mask_with_datetimeindex_tz(self, tz):
        mask = np.array([True, False, True, False])
        idx = date_range('20010101', periods=4, tz=tz)
        df = DataFrame({'a': np.arange(4)}, index=idx).astype('float64')
        result = df.copy()
        result.loc[mask, :] = df.loc[mask, :]
        tm.assert_frame_equal(result, df)
        result = df.copy()
        result.loc[mask] = df.loc[mask]
        tm.assert_frame_equal(result, df)

    def test_loc_setitem_mask_and_label_with_datetimeindex(self):
        df = DataFrame(np.arange(6.0).reshape(3, 2), columns=list('AB'), index=date_range('1/1/2000', periods=3, freq='1h'))
        expected = df.copy()
        expected['C'] = [expected.index[0]] + [pd.NaT, pd.NaT]
        mask = df.A < 1
        df.loc[mask, 'C'] = df.loc[mask].index
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_mask_td64_series_value(self):
        td1 = Timedelta(0)
        td2 = Timedelta(28767471428571405)
        df = DataFrame({'col': Series([td1, td2])})
        df_copy = df.copy()
        ser = Series([td1])
        expected = df['col'].iloc[1]._value
        df.loc[[True, False]] = ser
        result = df['col'].iloc[1]._value
        assert expected == result
        tm.assert_frame_equal(df, df_copy)

    @td.skip_array_manager_invalid_test
    def test_loc_setitem_boolean_and_column(self, float_frame):
        expected = float_frame.copy()
        mask = float_frame['A'] > 0
        float_frame.loc[mask, 'B'] = 0
        values = expected.values.copy()
        values[mask.values, 1] = 0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(float_frame, expected)

    def test_loc_setitem_ndframe_values_alignment(self, using_copy_on_write, warn_copy_on_write):
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], ['a']] = DataFrame({'a': [10, 20, 30]}, index=[2, 1, 0])
        expected = DataFrame({'a': [1, 2, 10], 'b': [4, 5, 6]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], ['a']] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], 'a'] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df_orig = df.copy()
        ser = df['a']
        with tm.assert_cow_warning(warn_copy_on_write):
            ser.loc[[False, False, True]] = Series([10, 11, 12], index=[2, 1, 0])
        if using_copy_on_write:
            tm.assert_frame_equal(df, df_orig)
        else:
            tm.assert_frame_equal(df, expected)

    def test_loc_indexer_empty_broadcast(self):
        df = DataFrame({'a': [], 'b': []}, dtype=object)
        expected = df.copy()
        df.loc[np.array([], dtype=np.bool_), ['a']] = df['a'].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_all_false_broadcast(self):
        df = DataFrame({'a': ['x'], 'b': ['y']}, dtype=object)
        expected = df.copy()
        df.loc[np.array([False], dtype=np.bool_), ['a']] = df['b'].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_length_one(self):
        df = DataFrame({'a': ['x'], 'b': ['y']}, dtype=object)
        expected = DataFrame({'a': ['y'], 'b': ['y']}, dtype=object)
        df.loc[np.array([True], dtype=np.bool_), ['a']] = df['b'].copy()
        tm.assert_frame_equal(df, expected)