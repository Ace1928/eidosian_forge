import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
class TestDataframeNoneCoercion:
    EXPECTED_SINGLE_ROW_RESULTS = [([1, 2, 3], [np.nan, 2, 3], FutureWarning), ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], None), ([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)], None), (['foo', 'bar', 'baz'], [None, 'bar', 'baz'], None)]

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_loc(self, expected):
        start_data, expected_result, warn = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe.loc[0, ['foo']] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_setitem_and_dataframe(self, expected):
        start_data, expected_result, warn = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe[start_dataframe['foo'] == start_dataframe['foo'][0]] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_none_coercion_loc_and_dataframe(self, expected):
        start_data, expected_result, warn = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe.loc[start_dataframe['foo'] == start_dataframe['foo'][0]] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    def test_none_coercion_mixed_dtypes(self):
        start_dataframe = DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': ['a', 'b', 'c']})
        start_dataframe.iloc[0] = None
        exp = DataFrame({'a': [np.nan, 2, 3], 'b': [np.nan, 2.0, 3.0], 'c': [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': [None, 'b', 'c']})
        tm.assert_frame_equal(start_dataframe, exp)