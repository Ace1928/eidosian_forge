from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex:
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
    if len(columns.levels) <= 2:
        return columns.levels[0]._rename(name=columns.names[0])
    levs = [[lev[c] if c >= 0 else None for c in codes] for lev, codes in zip(columns.levels[:-1], columns.codes[:-1])]
    tuples = zip(*levs)
    unique_tuples = (key for key, _ in itertools.groupby(tuples))
    new_levs = zip(*unique_tuples)
    return MultiIndex.from_arrays([Index(new_lev, dtype=lev.dtype) if None not in new_lev else new_lev for new_lev, lev in zip(new_levs, columns.levels)], names=columns.names[:-1])