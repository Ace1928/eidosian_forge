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
@pytest.mark.parametrize('val', [1, 1.1, 1 + 1j, True])
@pytest.mark.parametrize('exp_dtype', [object])
class TestCoercionObject(CoercionTest):

    @pytest.fixture
    def obj(self):
        return Series(['a', 'b', 'c', 'd'], dtype=object)

    @pytest.fixture
    def warn(self):
        return None