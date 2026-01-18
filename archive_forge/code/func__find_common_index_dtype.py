from __future__ import annotations
import textwrap
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.cast import find_common_type
from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
def _find_common_index_dtype(inds):
    """
        Finds a common type for the indexes to pass through to resulting index.

        Parameters
        ----------
        inds: list of Index or list objects

        Returns
        -------
        The common type or None if no indexes were given
        """
    dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
    if dtypes:
        dtype = find_common_type(dtypes)
    else:
        dtype = None
    return dtype