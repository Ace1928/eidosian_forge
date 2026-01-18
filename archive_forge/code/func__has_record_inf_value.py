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
def _has_record_inf_value(record_as_array: np.ndarray) -> np.bool_:
    is_inf_in_record = np.zeros(len(record_as_array), dtype=bool)
    for i, value in enumerate(record_as_array):
        is_element_inf = False
        try:
            is_element_inf = np.isinf(value)
        except TypeError:
            is_element_inf = False
        is_inf_in_record[i] = is_element_inf
    return np.any(is_inf_in_record)