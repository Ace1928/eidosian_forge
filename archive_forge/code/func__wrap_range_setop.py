from __future__ import annotations
from abc import (
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta
def _wrap_range_setop(self, other, res_i8) -> Self:
    new_freq = None
    if not len(res_i8):
        new_freq = self.freq
    elif isinstance(res_i8, RangeIndex):
        new_freq = to_offset(Timedelta(res_i8.step))
    res_values = res_i8.values.view(self._data._ndarray.dtype)
    result = type(self._data)._simple_new(res_values, dtype=self.dtype, freq=new_freq)
    return cast('Self', self._wrap_setop_result(other, result))