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
@pytest.mark.parametrize('val,exp_dtype,warn', [(1, np.complex128, None), (1.1, np.complex128, None), (1 + 1j, np.complex128, None), (True, object, FutureWarning)])
class TestCoercionComplex(CoercionTest):

    @pytest.fixture
    def obj(self):
        return Series([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])