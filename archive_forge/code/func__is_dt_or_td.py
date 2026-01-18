from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit
def _is_dt_or_td(dtype: DtypeObj) -> bool:
    return isinstance(dtype, DatetimeTZDtype) or lib.is_np_dtype(dtype, 'mM')