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
def _encode_strings(self) -> None:
    """
        Encode strings in dta-specific encoding

        Do not encode columns marked for date conversion or for strL
        conversion. The strL converter independently handles conversion and
        also accepts empty string arrays.
        """
    convert_dates = self._convert_dates
    convert_strl = getattr(self, '_convert_strl', [])
    for i, col in enumerate(self.data):
        if i in convert_dates or col in convert_strl:
            continue
        column = self.data[col]
        dtype = column.dtype
        if dtype.type is np.object_:
            inferred_dtype = infer_dtype(column, skipna=True)
            if not (inferred_dtype == 'string' or len(column) == 0):
                col = column.name
                raise ValueError(f'Column `{col}` cannot be exported.\n\nOnly string-like object arrays\ncontaining all strings or a mix of strings and None can be exported.\nObject arrays containing only null values are prohibited. Other object\ntypes cannot be exported and must first be converted to one of the\nsupported types.')
            encoded = self.data[col].str.encode(self._encoding)
            if max_len_string_array(ensure_object(encoded._values)) <= self._max_string_length:
                self.data[col] = encoded