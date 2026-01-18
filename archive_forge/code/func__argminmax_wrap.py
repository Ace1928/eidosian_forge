from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def _argminmax_wrap(self, value, axis=None, func=None):
    res = func(value, axis)
    nans = np.min(value, axis)
    nullnan = isna(nans)
    if res.ndim:
        res[nullnan] = -1
    elif hasattr(nullnan, 'all') and nullnan.all() or (not hasattr(nullnan, 'all') and nullnan):
        res = -1
    return res