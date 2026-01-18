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
def _rank_calc(self, *, axis: AxisInt=0, method: str='average', na_option: str='keep', ascending: bool=True, pct: bool=False):
    if axis != 0:
        ranked = super()._rank(axis=axis, method=method, na_option=na_option, ascending=ascending, pct=pct)
        if method == 'average' or pct:
            pa_type = pa.float64()
        else:
            pa_type = pa.uint64()
        result = pa.array(ranked, type=pa_type, from_pandas=True)
        return result
    data = self._pa_array.combine_chunks()
    sort_keys = 'ascending' if ascending else 'descending'
    null_placement = 'at_start' if na_option == 'top' else 'at_end'
    tiebreaker = 'min' if method == 'average' else method
    result = pc.rank(data, sort_keys=sort_keys, null_placement=null_placement, tiebreaker=tiebreaker)
    if na_option == 'keep':
        mask = pc.is_null(self._pa_array)
        null = pa.scalar(None, type=result.type)
        result = pc.if_else(mask, null, result)
    if method == 'average':
        result_max = pc.rank(data, sort_keys=sort_keys, null_placement=null_placement, tiebreaker='max')
        result_max = result_max.cast(pa.float64())
        result_min = result.cast(pa.float64())
        result = pc.divide(pc.add(result_min, result_max), 2)
    if pct:
        if not pa.types.is_floating(result.type):
            result = result.cast(pa.float64())
        if method == 'dense':
            divisor = pc.max(result)
        else:
            divisor = pc.count(result)
        result = pc.divide(result, divisor)
    return result