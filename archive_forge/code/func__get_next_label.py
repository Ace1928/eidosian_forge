from __future__ import annotations
from operator import (
import textwrap
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.tslibs import (
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import (
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
def _get_next_label(label):
    dtype = getattr(label, 'dtype', type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = 'datetime64[ns]'
    dtype = pandas_dtype(dtype)
    if lib.is_np_dtype(dtype, 'mM') or isinstance(dtype, DatetimeTZDtype):
        return label + np.timedelta64(1, 'ns')
    elif is_integer_dtype(dtype):
        return label + 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.inf)
    else:
        raise TypeError(f'cannot determine next label for type {repr(type(label))}')