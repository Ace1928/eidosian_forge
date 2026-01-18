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
def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs):
    if isinstance(self.dtype, StringDtype):
        return super()._groupby_op(how=how, has_dropped_na=has_dropped_na, min_count=min_count, ngroups=ngroups, ids=ids, **kwargs)
    values: ExtensionArray
    pa_type = self._pa_array.type
    if pa.types.is_timestamp(pa_type):
        values = self._to_datetimearray()
    elif pa.types.is_duration(pa_type):
        values = self._to_timedeltaarray()
    else:
        values = self._to_masked()
    result = values._groupby_op(how=how, has_dropped_na=has_dropped_na, min_count=min_count, ngroups=ngroups, ids=ids, **kwargs)
    if isinstance(result, np.ndarray):
        return result
    return type(self)._from_sequence(result, copy=False)