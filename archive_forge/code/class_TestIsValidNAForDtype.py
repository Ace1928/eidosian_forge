from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestIsValidNAForDtype:

    def test_is_valid_na_for_dtype_interval(self):
        dtype = IntervalDtype('int64', 'left')
        assert not is_valid_na_for_dtype(NaT, dtype)
        dtype = IntervalDtype('datetime64[ns]', 'both')
        assert not is_valid_na_for_dtype(NaT, dtype)

    def test_is_valid_na_for_dtype_categorical(self):
        dtype = CategoricalDtype(categories=[0, 1, 2])
        assert is_valid_na_for_dtype(np.nan, dtype)
        assert not is_valid_na_for_dtype(NaT, dtype)
        assert not is_valid_na_for_dtype(np.datetime64('NaT', 'ns'), dtype)
        assert not is_valid_na_for_dtype(np.timedelta64('NaT', 'ns'), dtype)