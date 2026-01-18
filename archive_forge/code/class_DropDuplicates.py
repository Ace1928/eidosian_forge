import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class DropDuplicates:

    def test_drop_duplicates_metadata(self, idx):
        result = idx.drop_duplicates()
        tm.assert_index_equal(idx, result)
        assert idx.freq == result.freq
        idx_dup = idx.append(idx)
        result = idx_dup.drop_duplicates()
        expected = idx
        if not isinstance(idx, PeriodIndex):
            assert idx_dup.freq is None
            assert result.freq is None
            expected = idx._with_freq(None)
        else:
            assert result.freq == expected.freq
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('keep, expected, index', [('first', np.concatenate(([False] * 10, [True] * 5)), np.arange(0, 10, dtype=np.int64)), ('last', np.concatenate(([True] * 5, [False] * 10)), np.arange(5, 15, dtype=np.int64)), (False, np.concatenate(([True] * 5, [False] * 5, [True] * 5)), np.arange(5, 10, dtype=np.int64))])
    def test_drop_duplicates(self, keep, expected, index, idx):
        idx = idx.append(idx[:5])
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected)
        expected = idx[~expected]
        result = idx.drop_duplicates(keep=keep)
        tm.assert_index_equal(result, expected)
        result = Series(idx).drop_duplicates(keep=keep)
        expected = Series(expected, index=index)
        tm.assert_series_equal(result, expected)