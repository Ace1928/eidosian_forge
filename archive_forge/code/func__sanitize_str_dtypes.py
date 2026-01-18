from __future__ import annotations
from collections.abc import Sequence
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
def _sanitize_str_dtypes(result: np.ndarray, data, dtype: np.dtype | None, copy: bool) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """
    if issubclass(result.dtype.type, str):
        if not lib.is_scalar(data):
            if not np.all(isna(data)):
                data = np.array(data, dtype=dtype, copy=False)
            result = np.array(data, dtype=object, copy=copy)
    return result