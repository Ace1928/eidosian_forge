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
def _reduce_calc(self, name: str, *, skipna: bool=True, keepdims: bool=False, **kwargs):
    pa_result = self._reduce_pyarrow(name, skipna=skipna, **kwargs)
    if keepdims:
        if isinstance(pa_result, pa.Scalar):
            result = pa.array([pa_result.as_py()], type=pa_result.type)
        else:
            result = pa.array([pa_result], type=to_pyarrow_type(infer_dtype_from_scalar(pa_result)[0]))
        return result
    if pc.is_null(pa_result).as_py():
        return self.dtype.na_value
    elif isinstance(pa_result, pa.Scalar):
        return pa_result.as_py()
    else:
        return pa_result