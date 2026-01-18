from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('val,exp_dtype,warn', [(1, np.float32, None), pytest.param(1.1, np.float32, None, marks=pytest.mark.xfail(not np_version_gte1p24 or (np_version_gte1p24 and np._get_promotion_state() != 'weak'), reason='np.float32(1.1) ends up as 1.100000023841858, so np_can_hold_element raises and we cast to float64')), (1 + 1j, np.complex128, FutureWarning), (True, object, FutureWarning), (np.uint8(2), np.float32, None), (np.uint32(2), np.float32, None), (np.uint32(np.iinfo(np.uint32).max), np.float64, FutureWarning), (np.uint64(2), np.float32, None), (np.int64(2), np.float32, None)])
class TestCoercionFloat32(CoercionTest):

    @pytest.fixture
    def obj(self):
        return Series([1.1, 2.2, 3.3, 4.4], dtype=np.float32)

    def test_slice_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
        super().test_slice_key(obj, key, expected, warn, val, indexer_sli, is_inplace)
        if isinstance(val, float):
            raise AssertionError('xfail not relevant for this test.')