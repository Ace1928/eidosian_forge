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
def _to_datetimearray(self) -> DatetimeArray:
    """Convert a pyarrow timestamp typed array to a DatetimeArray."""
    from pandas.core.arrays.datetimes import DatetimeArray, tz_to_dtype
    pa_type = self._pa_array.type
    assert pa.types.is_timestamp(pa_type)
    np_dtype = np.dtype(f'M8[{pa_type.unit}]')
    dtype = tz_to_dtype(pa_type.tz, pa_type.unit)
    np_array = self._pa_array.to_numpy()
    np_array = np_array.astype(np_dtype)
    return DatetimeArray._simple_new(np_array, dtype=dtype)