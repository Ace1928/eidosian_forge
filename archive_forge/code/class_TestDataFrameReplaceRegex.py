from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameReplaceRegex:

    @pytest.mark.parametrize('data', [{'a': list('ab..'), 'b': list('efgh')}, {'a': list('ab..'), 'b': list(range(4))}])
    @pytest.mark.parametrize('to_replace,value', [('\\s*\\.\\s*', np.nan), ('\\s*(\\.)\\s*', '\\1\\1\\1')])
    @pytest.mark.parametrize('compile_regex', [True, False])
    @pytest.mark.parametrize('regex_kwarg', [True, False])
    @pytest.mark.parametrize('inplace', [True, False])
    def test_regex_replace_scalar(self, data, to_replace, value, compile_regex, regex_kwarg, inplace):
        df = DataFrame(data)
        expected = df.copy()
        if compile_regex:
            to_replace = re.compile(to_replace)
        if regex_kwarg:
            regex = to_replace
            to_replace = None
        else:
            regex = True
        result = df.replace(to_replace, value, inplace=inplace, regex=regex)
        if inplace:
            assert result is None
            result = df
        if value is np.nan:
            expected_replace_val = np.nan
        else:
            expected_replace_val = '...'
        expected.loc[expected['a'] == '.', 'a'] = expected_replace_val
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set float into string")
    @pytest.mark.parametrize('regex', [False, True])
    def test_replace_regex_dtype_frame(self, regex):
        df1 = DataFrame({'A': ['0'], 'B': ['0']})
        expected_df1 = DataFrame({'A': [1], 'B': [1]})
        msg = 'Downcasting behavior in `replace`'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result_df1 = df1.replace(to_replace='0', value=1, regex=regex)
        tm.assert_frame_equal(result_df1, expected_df1)
        df2 = DataFrame({'A': ['0'], 'B': ['1']})
        expected_df2 = DataFrame({'A': [1], 'B': ['1']})
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result_df2 = df2.replace(to_replace='0', value=1, regex=regex)
        tm.assert_frame_equal(result_df2, expected_df2)

    def test_replace_with_value_also_being_replaced(self):
        df = DataFrame({'A': [0, 1, 2], 'B': [1, 0, 2]})
        result = df.replace({0: 1, 1: np.nan})
        expected = DataFrame({'A': [1, np.nan, 2], 'B': [np.nan, 1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_replace_categorical_no_replacement(self):
        df = DataFrame({'a': ['one', 'two', None, 'three'], 'b': ['one', None, 'two', 'three']}, dtype='category')
        expected = df.copy()
        result = df.replace(to_replace=['.', 'def'], value=['_', None])
        tm.assert_frame_equal(result, expected)

    def test_replace_object_splitting(self, using_infer_string):
        df = DataFrame({'a': ['a'], 'b': 'b'})
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1
        df.replace(to_replace='^\\s*$', value='', inplace=True, regex=True)
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1