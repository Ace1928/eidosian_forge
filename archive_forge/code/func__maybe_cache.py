from __future__ import annotations
from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.parsing import (
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.arrays import (
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import (
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
def _maybe_cache(arg: ArrayConvertible, format: str | None, cache: bool, convert_listlike: Callable) -> Series:
    """
    Create a cache of unique dates from an array of dates

    Parameters
    ----------
    arg : listlike, tuple, 1-d array, Series
    format : string
        Strftime format to parse time
    cache : bool
        True attempts to create a cache of converted values
    convert_listlike : function
        Conversion function to apply on dates

    Returns
    -------
    cache_array : Series
        Cache of converted, unique dates. Can be empty
    """
    from pandas import Series
    cache_array = Series(dtype=object)
    if cache:
        if not should_cache(arg):
            return cache_array
        if not isinstance(arg, (np.ndarray, ExtensionArray, Index, ABCSeries)):
            arg = np.array(arg)
        unique_dates = unique(arg)
        if len(unique_dates) < len(arg):
            cache_dates = convert_listlike(unique_dates, format)
            try:
                cache_array = Series(cache_dates, index=unique_dates, copy=False)
            except OutOfBoundsDatetime:
                return cache_array
            if not cache_array.index.is_unique:
                cache_array = cache_array[~cache_array.index.duplicated()]
    return cache_array