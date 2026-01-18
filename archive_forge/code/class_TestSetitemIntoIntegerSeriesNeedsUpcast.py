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
@pytest.mark.parametrize('val', [512, np.int16(512)])
class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):

    @pytest.fixture
    def obj(self):
        return Series([1, 2, 3], dtype=np.int8)

    @pytest.fixture
    def key(self):
        return 1

    @pytest.fixture
    def expected(self):
        return Series([1, 512, 3], dtype=np.int16)

    @pytest.fixture
    def warn(self):
        return FutureWarning