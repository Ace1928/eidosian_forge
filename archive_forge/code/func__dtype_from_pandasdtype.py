from __future__ import annotations
from typing import Any
import numpy as np
from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.core.interchange.buffer import PandasBuffer
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def _dtype_from_pandasdtype(self, dtype) -> tuple[DtypeKind, int, str, str]:
    """
        See `self.dtype` for details.
        """
    kind = _NP_KINDS.get(dtype.kind, None)
    if kind is None:
        raise ValueError(f'Data type {dtype} not supported by interchange protocol')
    if isinstance(dtype, ArrowDtype):
        byteorder = dtype.numpy_dtype.byteorder
    elif isinstance(dtype, DatetimeTZDtype):
        byteorder = dtype.base.byteorder
    elif isinstance(dtype, BaseMaskedDtype):
        byteorder = dtype.numpy_dtype.byteorder
    else:
        byteorder = dtype.byteorder
    return (kind, dtype.itemsize * 8, dtype_to_arrow_c_fmt(dtype), byteorder)