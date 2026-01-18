from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
def _construct_from_dt64_naive(data: np.ndarray, *, tz: tzinfo | None, copy: bool, ambiguous: TimeAmbiguous) -> tuple[np.ndarray, bool]:
    """
    Convert datetime64 data to a supported dtype, localizing if necessary.
    """
    new_dtype = data.dtype
    if not is_supported_dtype(new_dtype):
        new_dtype = get_supported_dtype(new_dtype)
        data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
        copy = False
    if data.dtype.byteorder == '>':
        data = data.astype(data.dtype.newbyteorder('<'))
        new_dtype = data.dtype
        copy = False
    if tz is not None:
        shape = data.shape
        if data.ndim > 1:
            data = data.ravel()
        data_unit = get_unit_from_dtype(new_dtype)
        data = tzconversion.tz_localize_to_utc(data.view('i8'), tz, ambiguous=ambiguous, creso=data_unit)
        data = data.view(new_dtype)
        data = data.reshape(shape)
    assert data.dtype == new_dtype, data.dtype
    result = data
    return (result, copy)