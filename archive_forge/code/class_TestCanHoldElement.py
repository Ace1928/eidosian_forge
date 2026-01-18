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
class TestCanHoldElement:

    @pytest.fixture(params=[lambda x: x, lambda x: x.to_series(), lambda x: x._data, lambda x: list(x), lambda x: x.astype(object), lambda x: np.asarray(x), lambda x: x[0], lambda x: x[:0]])
    def element(self, request):
        """
        Functions that take an Index and return an element that should have
        blk._can_hold_element(element) for a Block with this index's dtype.
        """
        return request.param

    def test_datetime_block_can_hold_element(self):
        block = create_block('datetime', [0])
        assert block._can_hold_element([])
        arr = pd.array(block.values.ravel())
        assert block._can_hold_element(None)
        arr[0] = None
        assert arr[0] is pd.NaT
        vals = [np.datetime64('2010-10-10'), datetime(2010, 10, 10)]
        for val in vals:
            assert block._can_hold_element(val)
            arr[0] = val
        val = date(2010, 10, 10)
        assert not block._can_hold_element(val)
        msg = "value should be a 'Timestamp', 'NaT', or array of those. Got 'date' instead."
        with pytest.raises(TypeError, match=msg):
            arr[0] = val

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element_emptylist(self, dtype, element):
        arr = np.array([1, 3, 4], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)
        assert blk._can_hold_element([])

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element(self, dtype, element):
        arr = np.array([1, 3, 4, 9], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)
        elem = element(ii)
        self.check_series_setitem(elem, ii, True)
        assert blk._can_hold_element(elem)
        ii2 = IntervalIndex.from_breaks(arr[:-1], closed='neither')
        elem = element(ii2)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)
        ii3 = IntervalIndex.from_breaks([Timestamp(1), Timestamp(3), Timestamp(4)])
        elem = element(ii3)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)
        ii4 = IntervalIndex.from_breaks([Timedelta(1), Timedelta(3), Timedelta(4)])
        elem = element(ii4)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

    def test_period_can_hold_element_emptylist(self):
        pi = period_range('2016', periods=3, freq='Y')
        blk = new_block(pi._data.reshape(1, 3), BlockPlacement([1]), ndim=2)
        assert blk._can_hold_element([])

    def test_period_can_hold_element(self, element):
        pi = period_range('2016', periods=3, freq='Y')
        elem = element(pi)
        self.check_series_setitem(elem, pi, True)
        pi2 = pi.asfreq('D')[:-1]
        elem = element(pi2)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, pi, False)
        dti = pi.to_timestamp('s')[:-1]
        elem = element(dti)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, pi, False)

    def check_can_hold_element(self, obj, elem, inplace: bool):
        blk = obj._mgr.blocks[0]
        if inplace:
            assert blk._can_hold_element(elem)
        else:
            assert not blk._can_hold_element(elem)

    def check_series_setitem(self, elem, index: Index, inplace: bool):
        arr = index._data.copy()
        ser = Series(arr, copy=False)
        self.check_can_hold_element(ser, elem, inplace)
        if is_scalar(elem):
            ser[0] = elem
        else:
            ser[:len(elem)] = elem
        if inplace:
            assert ser.array is arr
        else:
            assert ser.dtype == object