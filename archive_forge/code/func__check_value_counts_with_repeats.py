import numpy as np
from pandas import (
import pandas._testing as tm
def _check_value_counts_with_repeats(self, orig):
    idx = type(orig)(np.repeat(orig._values, range(1, len(orig) + 1)), dtype=orig.dtype)
    exp_idx = orig[::-1]
    if not isinstance(exp_idx, PeriodIndex):
        exp_idx = exp_idx._with_freq(None)
    expected = Series(range(10, 0, -1), index=exp_idx, dtype='int64', name='count')
    for obj in [idx, Series(idx)]:
        tm.assert_series_equal(obj.value_counts(), expected)
    tm.assert_index_equal(idx.unique(), orig)