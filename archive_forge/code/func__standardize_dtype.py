from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import (
from pandas.core.arrays.masked import (
@classmethod
def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
    """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
    if isinstance(dtype, str) and dtype.startswith(('Int', 'UInt', 'Float')):
        dtype = dtype.lower()
    if not isinstance(dtype, NumericDtype):
        mapping = cls._get_dtype_mapping()
        try:
            dtype = mapping[np.dtype(dtype)]
        except KeyError as err:
            raise ValueError(f'invalid dtype specified {dtype}') from err
    return dtype