from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
@final
def _set_noconvert_dtype_columns(self, col_indices: list[int], names: Sequence[Hashable]) -> set[int]:
    """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
    usecols: list[int] | list[str] | None
    noconvert_columns = set()
    if self.usecols_dtype == 'integer':
        usecols = sorted(self.usecols)
    elif callable(self.usecols) or self.usecols_dtype not in ('empty', None):
        usecols = col_indices
    else:
        usecols = None

    def _set(x) -> int:
        if usecols is not None and is_integer(x):
            x = usecols[x]
        if not is_integer(x):
            x = col_indices[names.index(x)]
        return x
    if isinstance(self.parse_dates, list):
        for val in self.parse_dates:
            if isinstance(val, list):
                for k in val:
                    noconvert_columns.add(_set(k))
            else:
                noconvert_columns.add(_set(val))
    elif isinstance(self.parse_dates, dict):
        for val in self.parse_dates.values():
            if isinstance(val, list):
                for k in val:
                    noconvert_columns.add(_set(k))
            else:
                noconvert_columns.add(_set(val))
    elif self.parse_dates:
        if isinstance(self.index_col, list):
            for k in self.index_col:
                noconvert_columns.add(_set(k))
        elif self.index_col is not None:
            noconvert_columns.add(_set(self.index_col))
    return noconvert_columns