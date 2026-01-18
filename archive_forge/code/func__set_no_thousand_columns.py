from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _set_no_thousand_columns(self) -> set[int]:
    no_thousands_columns: set[int] = set()
    if self.columns and self.parse_dates:
        assert self._col_indices is not None
        no_thousands_columns = self._set_noconvert_dtype_columns(self._col_indices, self.columns)
    if self.columns and self.dtype:
        assert self._col_indices is not None
        for i, col in zip(self._col_indices, self.columns):
            if not isinstance(self.dtype, dict) and (not is_numeric_dtype(self.dtype)):
                no_thousands_columns.add(i)
            if isinstance(self.dtype, dict) and col in self.dtype and (not is_numeric_dtype(self.dtype[col]) or is_bool_dtype(self.dtype[col])):
                no_thousands_columns.add(i)
    return no_thousands_columns