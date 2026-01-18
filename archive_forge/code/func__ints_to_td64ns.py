from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.timedeltas import (
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer
import textwrap
def _ints_to_td64ns(data, unit: str='ns'):
    """
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    """
    copy_made = False
    unit = unit if unit is not None else 'ns'
    if data.dtype != np.int64:
        data = data.astype(np.int64)
        copy_made = True
    if unit != 'ns':
        dtype_str = f'timedelta64[{unit}]'
        data = data.view(dtype_str)
        data = astype_overflowsafe(data, dtype=TD64NS_DTYPE)
        copy_made = True
    else:
        data = data.view('timedelta64[ns]')
    return (data, copy_made)