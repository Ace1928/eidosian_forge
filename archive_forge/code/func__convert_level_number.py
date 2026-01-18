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
def _convert_level_number(level_num: int, columns: Index):
    """
        Logic for converting the level number to something we can safely pass
        to swaplevel.

        If `level_num` matches a column name return the name from
        position `level_num`, otherwise return `level_num`.
        """
    if level_num in columns.names:
        return columns.names[level_num]
    return level_num