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
@final
def _aggregate_series_pure_python(self, obj: Series, func: Callable) -> npt.NDArray[np.object_]:
    _, _, ngroups = self.group_info
    result = np.empty(ngroups, dtype='O')
    initialized = False
    splitter = self._get_splitter(obj, axis=0)
    for i, group in enumerate(splitter):
        res = func(group)
        res = extract_result(res)
        if not initialized:
            check_result_array(res, group.dtype)
            initialized = True
        result[i] = res
    return result