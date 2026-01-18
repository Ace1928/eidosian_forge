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
def _box_pa_array(cls, value, pa_type: pa.DataType | None=None, copy: bool=False) -> pa.Array | pa.ChunkedArray:
    """
        Box value into a pyarrow Array or ChunkedArray.

        Parameters
        ----------
        value : Sequence
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
    if isinstance(value, cls):
        pa_array = value._pa_array
    elif isinstance(value, (pa.Array, pa.ChunkedArray)):
        pa_array = value
    elif isinstance(value, BaseMaskedArray):
        if copy:
            value = value.copy()
        pa_array = value.__arrow_array__()
    else:
        if isinstance(value, np.ndarray) and pa_type is not None and (pa.types.is_large_binary(pa_type) or pa.types.is_large_string(pa_type)):
            value = value.tolist()
        elif copy and is_array_like(value):
            value = value.copy()
        if pa_type is not None and pa.types.is_duration(pa_type) and (not isinstance(value, np.ndarray) or value.dtype.kind not in 'mi'):
            from pandas.core.tools.timedeltas import to_timedelta
            value = to_timedelta(value, unit=pa_type.unit).as_unit(pa_type.unit)
            value = value.to_numpy()
        try:
            pa_array = pa.array(value, type=pa_type, from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            pa_array = pa.array(value, from_pandas=True)
        if pa_type is None and pa.types.is_duration(pa_array.type):
            from pandas.core.tools.timedeltas import to_timedelta
            value = to_timedelta(value)
            value = value.to_numpy()
            pa_array = pa.array(value, type=pa_type, from_pandas=True)
        if pa.types.is_duration(pa_array.type) and pa_array.null_count > 0:
            arr = cls(pa_array)
            arr = arr.fillna(arr.dtype.na_value)
            pa_array = arr._pa_array
    if pa_type is not None and pa_array.type != pa_type:
        if pa.types.is_dictionary(pa_type):
            pa_array = pa_array.dictionary_encode()
        else:
            try:
                pa_array = pa_array.cast(pa_type)
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                if pa.types.is_string(pa_array.type) or pa.types.is_large_string(pa_array.type):
                    return cls._from_sequence_of_strings(value, dtype=pa_type)._pa_array
                else:
                    raise
    return pa_array