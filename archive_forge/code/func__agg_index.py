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
def _agg_index(self, index, try_parse_dates: bool=True) -> Index:
    arrays = []
    converters = self._clean_mapping(self.converters)
    for i, arr in enumerate(index):
        if try_parse_dates and self._should_parse_dates(i):
            arr = self._date_conv(arr, col=self.index_names[i] if self.index_names is not None else None)
        if self.na_filter:
            col_na_values = self.na_values
            col_na_fvalues = self.na_fvalues
        else:
            col_na_values = set()
            col_na_fvalues = set()
        if isinstance(self.na_values, dict):
            assert self.index_names is not None
            col_name = self.index_names[i]
            if col_name is not None:
                col_na_values, col_na_fvalues = _get_na_values(col_name, self.na_values, self.na_fvalues, self.keep_default_na)
        clean_dtypes = self._clean_mapping(self.dtype)
        cast_type = None
        index_converter = False
        if self.index_names is not None:
            if isinstance(clean_dtypes, dict):
                cast_type = clean_dtypes.get(self.index_names[i], None)
            if isinstance(converters, dict):
                index_converter = converters.get(self.index_names[i]) is not None
        try_num_bool = not (cast_type and is_string_dtype(cast_type) or index_converter)
        arr, _ = self._infer_types(arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool)
        arrays.append(arr)
    names = self.index_names
    index = ensure_index_from_sequences(arrays, names)
    return index