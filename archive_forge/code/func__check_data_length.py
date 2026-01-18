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
def _check_data_length(self, columns: Sequence[Hashable], data: Sequence[ArrayLike]) -> None:
    """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
    if not self.index_col and len(columns) != len(data) and columns:
        empty_str = is_object_dtype(data[-1]) and data[-1] == ''
        empty_str_or_na = empty_str | isna(data[-1])
        if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
            return
        warnings.warn('Length of header or names does not match length of data. This leads to a loss of data with index_col=False.', ParserWarning, stacklevel=find_stack_level())