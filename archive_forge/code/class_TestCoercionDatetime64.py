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
@pytest.mark.parametrize('val,exp_dtype,warn', [(Timestamp('2012-01-01'), 'datetime64[ns]', None), (1, object, FutureWarning), ('x', object, FutureWarning)])
class TestCoercionDatetime64(CoercionTest):

    @pytest.fixture
    def obj(self):
        return Series(date_range('2011-01-01', freq='D', periods=4))

    @pytest.fixture
    def warn(self):
        return None