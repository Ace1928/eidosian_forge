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
class TestSetitemDT64IntoInt(SetitemCastingEquivalents):

    @pytest.fixture(params=['M8[ns]', 'm8[ns]'])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def scalar(self, dtype):
        val = np.datetime64('2021-01-18 13:25:00', 'ns')
        if dtype == 'm8[ns]':
            val = val - val
        return val

    @pytest.fixture
    def expected(self, scalar):
        expected = Series([scalar, scalar, 3], dtype=object)
        assert isinstance(expected[0], type(scalar))
        return expected

    @pytest.fixture
    def obj(self):
        return Series([1, 2, 3])

    @pytest.fixture
    def key(self):
        return slice(None, -1)

    @pytest.fixture(params=[None, list, np.array])
    def val(self, scalar, request):
        box = request.param
        if box is None:
            return scalar
        return box([scalar, scalar])

    @pytest.fixture
    def warn(self):
        return FutureWarning