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
def _get_validity_buffer(self) -> tuple[PandasBuffer, Any]:
    """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte mask.
        """
    null, invalid = self.describe_null
    if self.dtype[0] == DtypeKind.STRING:
        buf = self._col.to_numpy()
        valid = invalid == 0
        invalid = not valid
        mask = np.zeros(shape=(len(buf),), dtype=np.bool_)
        for i, obj in enumerate(buf):
            mask[i] = valid if isinstance(obj, str) else invalid
        buffer = PandasBuffer(mask)
        dtype = (DtypeKind.BOOL, 8, ArrowCTypes.BOOL, Endianness.NATIVE)
        return (buffer, dtype)
    try:
        msg = f'{_NO_VALIDITY_BUFFER[null]} so does not have a separate mask'
    except KeyError:
        raise NotImplementedError('See self.describe_null')
    raise NoBufferPresent(msg)