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
def _get_data_buffer(self) -> tuple[PandasBuffer, Any]:
    """
        Return the buffer containing the data and the buffer's associated dtype.
        """
    if self.dtype[0] in (DtypeKind.INT, DtypeKind.UINT, DtypeKind.FLOAT, DtypeKind.BOOL, DtypeKind.DATETIME):
        if self.dtype[0] == DtypeKind.DATETIME and len(self.dtype[2]) > 4:
            np_arr = self._col.dt.tz_convert(None).to_numpy()
        else:
            np_arr = self._col.to_numpy()
        buffer = PandasBuffer(np_arr, allow_copy=self._allow_copy)
        dtype = self.dtype
    elif self.dtype[0] == DtypeKind.CATEGORICAL:
        codes = self._col.values._codes
        buffer = PandasBuffer(codes, allow_copy=self._allow_copy)
        dtype = self._dtype_from_pandasdtype(codes.dtype)
    elif self.dtype[0] == DtypeKind.STRING:
        buf = self._col.to_numpy()
        b = bytearray()
        for obj in buf:
            if isinstance(obj, str):
                b.extend(obj.encode(encoding='utf-8'))
        buffer = PandasBuffer(np.frombuffer(b, dtype='uint8'))
        dtype = self.dtype
    else:
        raise NotImplementedError(f'Data type {self._col.dtype} not handled yet')
    return (buffer, dtype)