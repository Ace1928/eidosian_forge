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
@classmethod
def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None=None, copy: bool=False):
    """
        Construct a new ExtensionArray from a sequence of strings.
        """
    pa_type = to_pyarrow_type(dtype)
    if pa_type is None or pa.types.is_binary(pa_type) or pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        scalars = strings
    elif pa.types.is_timestamp(pa_type):
        from pandas.core.tools.datetimes import to_datetime
        scalars = to_datetime(strings, errors='raise')
    elif pa.types.is_date(pa_type):
        from pandas.core.tools.datetimes import to_datetime
        scalars = to_datetime(strings, errors='raise').date
    elif pa.types.is_duration(pa_type):
        from pandas.core.tools.timedeltas import to_timedelta
        scalars = to_timedelta(strings, errors='raise')
        if pa_type.unit != 'ns':
            mask = isna(scalars)
            if not isinstance(strings, (pa.Array, pa.ChunkedArray)):
                strings = pa.array(strings, type=pa.string(), from_pandas=True)
            strings = pc.if_else(mask, None, strings)
            try:
                scalars = strings.cast(pa.int64())
            except pa.ArrowInvalid:
                pass
    elif pa.types.is_time(pa_type):
        from pandas.core.tools.times import to_time
        scalars = to_time(strings, errors='coerce')
    elif pa.types.is_boolean(pa_type):
        if isinstance(strings, (pa.Array, pa.ChunkedArray)):
            scalars = strings
        else:
            scalars = pa.array(strings, type=pa.string(), from_pandas=True)
        scalars = pc.if_else(pc.equal(scalars, '1.0'), '1', scalars)
        scalars = pc.if_else(pc.equal(scalars, '0.0'), '0', scalars)
        scalars = scalars.cast(pa.bool_())
    elif pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_decimal(pa_type):
        from pandas.core.tools.numeric import to_numeric
        scalars = to_numeric(strings, errors='raise')
    else:
        raise NotImplementedError(f'Converting strings to {pa_type} is not implemented.')
    return cls._from_sequence(scalars, dtype=pa_type, copy=copy)