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
@pytest.mark.parametrize('val', ['foo', Period('2016', freq='Y'), Interval(1, 2, closed='both')])
@pytest.mark.parametrize('exp_dtype', [object])
class TestPeriodIntervalCoercion(CoercionTest):

    @pytest.fixture(params=[period_range('2016-01-01', periods=3, freq='D'), interval_range(1, 5)])
    def obj(self, request):
        return Series(request.param)

    @pytest.fixture
    def warn(self):
        return FutureWarning