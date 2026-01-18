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
def _preprocess_for_cut(x) -> Index:
    """
    handles preprocessing for cut where we convert passed
    input to array, strip the index information and store it
    separately
    """
    ndim = getattr(x, 'ndim', None)
    if ndim is None:
        x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('Input array must be 1 dimensional')
    return Index(x)