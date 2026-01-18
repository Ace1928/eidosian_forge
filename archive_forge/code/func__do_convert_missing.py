from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame:
    replacements = {}
    for i in range(len(data.columns)):
        fmt = self._typlist[i]
        if fmt not in self.VALID_RANGE:
            continue
        fmt = cast(str, fmt)
        nmin, nmax = self.VALID_RANGE[fmt]
        series = data.iloc[:, i]
        svals = series._values
        missing = (svals < nmin) | (svals > nmax)
        if not missing.any():
            continue
        if convert_missing:
            missing_loc = np.nonzero(np.asarray(missing))[0]
            umissing, umissing_loc = np.unique(series[missing], return_inverse=True)
            replacement = Series(series, dtype=object)
            for j, um in enumerate(umissing):
                missing_value = StataMissingValue(um)
                loc = missing_loc[umissing_loc == j]
                replacement.iloc[loc] = missing_value
        else:
            dtype = series.dtype
            if dtype not in (np.float32, np.float64):
                dtype = np.float64
            replacement = Series(series, dtype=dtype)
            if not replacement._values.flags['WRITEABLE']:
                replacement = replacement.copy()
            replacement._values[missing] = np.nan
        replacements[i] = replacement
    if replacements:
        for idx, value in replacements.items():
            data.isetitem(idx, value)
    return data