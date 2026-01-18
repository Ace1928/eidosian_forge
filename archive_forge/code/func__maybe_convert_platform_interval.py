from __future__ import annotations
import operator
from operator import (
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.algorithms import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
from_arrays
from_tuples
from_breaks
def _maybe_convert_platform_interval(values) -> ArrayLike:
    """
    Try to do platform conversion, with special casing for IntervalArray.
    Wrapper around maybe_convert_platform that alters the default return
    dtype in certain cases to be compatible with IntervalArray.  For example,
    empty lists return with integer dtype instead of object dtype, which is
    prohibited for IntervalArray.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    array
    """
    if isinstance(values, (list, tuple)) and len(values) == 0:
        return np.array([], dtype=np.int64)
    elif not is_list_like(values) or isinstance(values, ABCDataFrame):
        return values
    elif isinstance(getattr(values, 'dtype', None), CategoricalDtype):
        values = np.asarray(values)
    elif not hasattr(values, 'dtype') and (not isinstance(values, (list, tuple, range))):
        return values
    else:
        values = extract_array(values, extract_numpy=True)
    if not hasattr(values, 'dtype'):
        values = np.asarray(values)
        if values.dtype.kind in 'iu' and values.dtype != np.int64:
            values = values.astype(np.int64)
    return values