from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
class TestBlockPlacement:

    @pytest.mark.parametrize('slc, expected', [(slice(0, 4), 4), (slice(0, 4, 2), 2), (slice(0, 3, 2), 2), (slice(0, 1, 2), 1), (slice(1, 0, -1), 1)])
    def test_slice_len(self, slc, expected):
        assert len(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize('slc', [slice(1, 1, 0), slice(1, 2, 0)])
    def test_zero_step_raises(self, slc):
        msg = 'slice step cannot be zero'
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    def test_slice_canonize_negative_stop(self):
        slc = slice(3, -1, -2)
        bp = BlockPlacement(slc)
        assert bp.indexer == slice(3, None, -2)

    @pytest.mark.parametrize('slc', [slice(None, None), slice(10, None), slice(None, None, -1), slice(None, 10, -1), slice(-1, None), slice(None, -1), slice(-1, -1), slice(-1, None, -1), slice(None, -1, -1), slice(-1, -1, -1)])
    def test_unbounded_slice_raises(self, slc):
        msg = 'unbounded slice'
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    @pytest.mark.parametrize('slc', [slice(0, 0), slice(100, 0), slice(100, 100), slice(100, 100, -1), slice(0, 100, -1)])
    def test_not_slice_like_slices(self, slc):
        assert not BlockPlacement(slc).is_slice_like

    @pytest.mark.parametrize('arr, slc', [([0], slice(0, 1, 1)), ([100], slice(100, 101, 1)), ([0, 1, 2], slice(0, 3, 1)), ([0, 5, 10], slice(0, 15, 5)), ([0, 100], slice(0, 200, 100)), ([2, 1], slice(2, 0, -1))])
    def test_array_to_slice_conversion(self, arr, slc):
        assert BlockPlacement(arr).as_slice == slc

    @pytest.mark.parametrize('arr', [[], [-1], [-1, -2, -3], [-10], [-1], [-1, 0, 1, 2], [-2, 0, 2, 4], [1, 0, -1], [1, 1, 1]])
    def test_not_slice_like_arrays(self, arr):
        assert not BlockPlacement(arr).is_slice_like

    @pytest.mark.parametrize('slc, expected', [(slice(0, 3), [0, 1, 2]), (slice(0, 0), []), (slice(3, 0), [])])
    def test_slice_iter(self, slc, expected):
        assert list(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize('slc, arr', [(slice(0, 3), [0, 1, 2]), (slice(0, 0), []), (slice(3, 0), []), (slice(3, 0, -1), [3, 2, 1])])
    def test_slice_to_array_conversion(self, slc, arr):
        tm.assert_numpy_array_equal(BlockPlacement(slc).as_array, np.asarray(arr, dtype=np.intp))

    def test_blockplacement_add(self):
        bpl = BlockPlacement(slice(0, 5))
        assert bpl.add(1).as_slice == slice(1, 6, 1)
        assert bpl.add(np.arange(5)).as_slice == slice(0, 10, 2)
        assert list(bpl.add(np.arange(5, 0, -1))) == [5, 5, 5, 5, 5]

    @pytest.mark.parametrize('val, inc, expected', [(slice(0, 0), 0, []), (slice(1, 4), 0, [1, 2, 3]), (slice(3, 0, -1), 0, [3, 2, 1]), ([1, 2, 4], 0, [1, 2, 4]), (slice(0, 0), 10, []), (slice(1, 4), 10, [11, 12, 13]), (slice(3, 0, -1), 10, [13, 12, 11]), ([1, 2, 4], 10, [11, 12, 14]), (slice(0, 0), -1, []), (slice(1, 4), -1, [0, 1, 2]), ([1, 2, 4], -1, [0, 1, 3])])
    def test_blockplacement_add_int(self, val, inc, expected):
        assert list(BlockPlacement(val).add(inc)) == expected

    @pytest.mark.parametrize('val', [slice(1, 4), [1, 2, 4]])
    def test_blockplacement_add_int_raises(self, val):
        msg = 'iadd causes length change'
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(val).add(-10)