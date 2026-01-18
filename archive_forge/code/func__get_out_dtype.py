from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
    how = self.how
    if how == 'rank':
        out_dtype = 'float64'
    elif how in ['idxmin', 'idxmax']:
        out_dtype = 'intp'
    elif dtype.kind in 'iufcb':
        out_dtype = f'{dtype.kind}{dtype.itemsize}'
    else:
        out_dtype = 'object'
    return np.dtype(out_dtype)