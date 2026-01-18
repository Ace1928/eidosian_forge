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
def _str_endswith(self, pat: str | tuple[str, ...], na=None):
    if isinstance(pat, str):
        result = pc.ends_with(self._pa_array, pattern=pat)
    elif len(pat) == 0:
        result = pc.if_else(pc.is_null(self._pa_array), None, False)
    else:
        result = pc.ends_with(self._pa_array, pattern=pat[0])
        for p in pat[1:]:
            result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p))
    if not isna(na):
        result = result.fill_null(na)
    return type(self)(result)