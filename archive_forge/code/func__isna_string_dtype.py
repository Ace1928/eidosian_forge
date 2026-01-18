from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def _isna_string_dtype(values: np.ndarray, inf_as_na: bool) -> npt.NDArray[np.bool_]:
    dtype = values.dtype
    if dtype.kind in ('S', 'U'):
        result = np.zeros(values.shape, dtype=bool)
    elif values.ndim in {1, 2}:
        result = libmissing.isnaobj(values, inf_as_na=inf_as_na)
    else:
        result = libmissing.isnaobj(values.ravel(), inf_as_na=inf_as_na)
        result = result.reshape(values.shape)
    return result