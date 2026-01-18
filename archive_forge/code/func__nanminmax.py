from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def _nanminmax(meth, fill_value_typ):

    @bottleneck_switch(name=f'nan{meth}')
    @_datetimelike_compat
    def reduction(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None):
        if values.size == 0:
            return _na_for_min_count(values, axis)
        values, mask = _get_values(values, skipna, fill_value_typ=fill_value_typ, mask=mask)
        result = getattr(values, meth)(axis)
        result = _maybe_null_out(result, axis, mask, values.shape)
        return result
    return reduction