import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestIntervalIndexInsideMultiIndex:

    def test_mi_intervalindex_slicing_with_scalar(self):
        ii = IntervalIndex.from_arrays([0, 1, 10, 11, 0, 1, 10, 11], [1, 2, 11, 12, 1, 2, 11, 12], name='MP')
        idx = pd.MultiIndex.from_arrays([pd.Index(['FC', 'FC', 'FC', 'FC', 'OWNER', 'OWNER', 'OWNER', 'OWNER']), pd.Index(['RID1', 'RID1', 'RID2', 'RID2', 'RID1', 'RID1', 'RID2', 'RID2']), ii])
        idx.names = ['Item', 'RID', 'MP']
        df = DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8]})
        df.index = idx
        query_df = DataFrame({'Item': ['FC', 'OWNER', 'FC', 'OWNER', 'OWNER'], 'RID': ['RID1', 'RID1', 'RID1', 'RID2', 'RID2'], 'MP': [0.2, 1.5, 1.6, 11.1, 10.9]})
        query_df = query_df.sort_index()
        idx = pd.MultiIndex.from_arrays([query_df.Item, query_df.RID, query_df.MP])
        query_df.index = idx
        result = df.value.loc[query_df.index]
        sliced_level = ii.take([0, 1, 1, 3, 2])
        expected_index = pd.MultiIndex.from_arrays([idx.get_level_values(0), idx.get_level_values(1), sliced_level])
        expected = Series([1, 6, 2, 8, 7], index=expected_index, name='value')
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(not IS64, reason='GH 23440')
    @pytest.mark.parametrize('base', [101, 1010])
    def test_reindex_behavior_with_interval_index(self, base):
        ser = Series(range(base), index=IntervalIndex.from_arrays(range(base), range(1, base + 1)))
        expected_result = Series([np.nan, 0], index=[np.nan, 1.0], dtype=float)
        result = ser.reindex(index=[np.nan, 1.0])
        tm.assert_series_equal(result, expected_result)