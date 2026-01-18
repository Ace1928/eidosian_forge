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
class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):

    @pytest.fixture(params=['m8[ns]', 'M8[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Central]'])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        i8vals = date_range('2016-01-01', periods=3).asi8
        idx = Index(i8vals, dtype=dtype)
        assert idx.dtype == dtype
        return Series(idx)

    @pytest.fixture(params=[None, np.nan, NaT, np.timedelta64('NaT', 'ns'), np.datetime64('NaT', 'ns')])
    def val(self, request):
        return request.param

    @pytest.fixture
    def is_inplace(self, val, obj):
        return val is NaT or val is None or val is np.nan or (obj.dtype == val.dtype)

    @pytest.fixture
    def expected(self, obj, val, is_inplace):
        dtype = obj.dtype if is_inplace else object
        expected = Series([val] + list(obj[1:]), dtype=dtype)
        return expected

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def warn(self, is_inplace):
        return None if is_inplace else FutureWarning