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
def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
    mgr = sdata._mgr.get_slice(slice_obj, axis=1 - self.axis)
    df = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
    return df.__finalize__(sdata, method='groupby')