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
def _infer_precision(base_precision: int, bins: Index) -> int:
    """
    Infer an appropriate precision for _round_frac
    """
    for precision in range(base_precision, 20):
        levels = np.asarray([_round_frac(b, precision) for b in bins])
        if algos.unique(levels).size == bins.size:
            return precision
    return base_precision