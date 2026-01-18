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
def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    """
    Return a list with distinct elements of "objs" (different ids).
    Preserves order.
    """
    ids: set[int] = set()
    res = []
    for obj in objs:
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res