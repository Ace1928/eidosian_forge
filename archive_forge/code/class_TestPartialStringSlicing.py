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
class TestPartialStringSlicing:

    def test_loc_getitem_partial_string_slicing_datetimeindex(self):
        df = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3]}, index=to_datetime(['2020-08-01', '2020-07-02', '2020-08-05']))
        expected = DataFrame({'col1': ['a', 'c'], 'col2': [1, 3]}, index=to_datetime(['2020-08-01', '2020-08-05']))
        result = df.loc['2020-08']
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(self):
        pi = pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
        ser = pi.to_series()
        result = ser.loc[:'2017-12']
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(self):
        ix = timedelta_range(start='1 day', end='2 days', freq='1h')
        ser = ix.to_series()
        result = ser.loc[:'1 days']
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self):
        df = DataFrame({'x': range(3)}, index=to_timedelta(range(3), unit='days'))
        expected = df.iloc[0]
        sliced = df.loc['0 days']
        tm.assert_series_equal(sliced, expected)

    @pytest.mark.parametrize('indexer_end', [None, '2020-01-02 23:59:59.999999999'])
    def test_loc_getitem_partial_slice_non_monotonicity(self, tz_aware_fixture, indexer_end, frame_or_series):
        obj = frame_or_series([1] * 5, index=DatetimeIndex([Timestamp('2019-12-30'), Timestamp('2020-01-01'), Timestamp('2019-12-25'), Timestamp('2020-01-02 23:59:59.999999999'), Timestamp('2019-12-19')], tz=tz_aware_fixture))
        expected = frame_or_series([1] * 2, index=DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02 23:59:59.999999999')], tz=tz_aware_fixture))
        indexer = slice('2020-01-01', indexer_end)
        result = obj[indexer]
        tm.assert_equal(result, expected)
        result = obj.loc[indexer]
        tm.assert_equal(result, expected)