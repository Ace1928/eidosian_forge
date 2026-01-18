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
def _dtype_to_stata_type_117(dtype: np.dtype, column: Series, force_strl: bool) -> int:
    """
    Converts dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 2045 are strings of this length
                Pandas    Stata
    32768 - for object    strL
    65526 - for int8      byte
    65527 - for int16     int
    65528 - for int32     long
    65529 - for float32   float
    65530 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
    if force_strl:
        return 32768
    if dtype.type is np.object_:
        itemsize = max_len_string_array(ensure_object(column._values))
        itemsize = max(itemsize, 1)
        if itemsize <= 2045:
            return itemsize
        return 32768
    elif dtype.type is np.float64:
        return 65526
    elif dtype.type is np.float32:
        return 65527
    elif dtype.type is np.int32:
        return 65528
    elif dtype.type is np.int16:
        return 65529
    elif dtype.type is np.int8:
        return 65530
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')