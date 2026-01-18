import numpy as np
from pandas import (
import pandas._testing as tm
def _check_value_counts_dropna(self, idx):
    exp_idx = idx[[2, 3]]
    expected = Series([3, 2], index=exp_idx, name='count')
    for obj in [idx, Series(idx)]:
        tm.assert_series_equal(obj.value_counts(), expected)
    exp_idx = idx[[2, 3, -1]]
    expected = Series([3, 2, 1], index=exp_idx, name='count')
    for obj in [idx, Series(idx)]:
        tm.assert_series_equal(obj.value_counts(dropna=False), expected)
    tm.assert_index_equal(idx.unique(), exp_idx)