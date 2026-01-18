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
def _convert_to_ndarrays(self, dct: Mapping, na_values, na_fvalues, verbose: bool=False, converters=None, dtypes=None):
    result = {}
    for c, values in dct.items():
        conv_f = None if converters is None else converters.get(c, None)
        if isinstance(dtypes, dict):
            cast_type = dtypes.get(c, None)
        else:
            cast_type = dtypes
        if self.na_filter:
            col_na_values, col_na_fvalues = _get_na_values(c, na_values, na_fvalues, self.keep_default_na)
        else:
            col_na_values, col_na_fvalues = (set(), set())
        if c in self._parse_date_cols:
            mask = algorithms.isin(values, set(col_na_values) | col_na_fvalues)
            np.putmask(values, mask, np.nan)
            result[c] = values
            continue
        if conv_f is not None:
            if cast_type is not None:
                warnings.warn(f'Both a converter and dtype were specified for column {c} - only the converter will be used.', ParserWarning, stacklevel=find_stack_level())
            try:
                values = lib.map_infer(values, conv_f)
            except ValueError:
                mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                values = lib.map_infer_mask(values, conv_f, mask)
            cvals, na_count = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool=False)
        else:
            is_ea = is_extension_array_dtype(cast_type)
            is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
            try_num_bool = not (cast_type and is_str_or_ea_dtype)
            cvals, na_count = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool)
            if cast_type is not None:
                cast_type = pandas_dtype(cast_type)
            if cast_type and (cvals.dtype != cast_type or is_ea):
                if not is_ea and na_count > 0:
                    if is_bool_dtype(cast_type):
                        raise ValueError(f'Bool column has NA values in column {c}')
                cvals = self._cast_types(cvals, cast_type, c)
        result[c] = cvals
        if verbose and na_count:
            print(f'Filled {na_count} NA values in column {c!s}')
    return result