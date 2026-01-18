from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
def _str_get_dummies(self, sep: str='|'):
    split = pc.split_pattern(self._pa_array, sep)
    flattened_values = pc.list_flatten(split)
    uniques = flattened_values.unique()
    uniques_sorted = uniques.take(pa.compute.array_sort_indices(uniques))
    lengths = pc.list_value_length(split).fill_null(0).to_numpy()
    n_rows = len(self)
    n_cols = len(uniques)
    indices = pc.index_in(flattened_values, uniques_sorted).to_numpy()
    indices = indices + np.arange(n_rows).repeat(lengths) * n_cols
    dummies = np.zeros(n_rows * n_cols, dtype=np.bool_)
    dummies[indices] = True
    dummies = dummies.reshape((n_rows, n_cols))
    result = type(self)(pa.array(list(dummies)))
    return (result, uniques_sorted.to_pylist())