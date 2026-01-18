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
def _convert_and_box_cache(arg: DatetimeScalarOrArrayConvertible, cache_array: Series, name: Hashable | None=None) -> Index:
    """
    Convert array of dates with a cache and wrap the result in an Index.

    Parameters
    ----------
    arg : integer, float, string, datetime, list, tuple, 1-d array, Series
    cache_array : Series
        Cache of converted, unique dates
    name : string, default None
        Name for a DatetimeIndex

    Returns
    -------
    result : Index-like of converted dates
    """
    from pandas import Series
    result = Series(arg, dtype=cache_array.index.dtype).map(cache_array)
    return _box_as_indexlike(result._values, utc=False, name=name)