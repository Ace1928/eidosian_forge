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
class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):

    @pytest.fixture
    def val(self):
        td = np.timedelta64(4, 'ns')
        return td

    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        arr = np.arange(5).astype(dtype)
        ser = Series(arr)
        return ser

    @pytest.fixture
    def expected(self, dtype):
        arr = np.arange(5).astype(dtype)
        ser = Series(arr)
        ser = ser.astype(object)
        ser.iloc[0] = np.timedelta64(4, 'ns')
        return ser

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def warn(self):
        return FutureWarning