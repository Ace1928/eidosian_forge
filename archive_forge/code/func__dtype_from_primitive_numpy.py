from math import ceil
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
from modin.utils import _inherit_docstrings
from .buffer import HdkProtocolBuffer
from .utils import arrow_dtype_to_arrow_c, arrow_types_map
def _dtype_from_primitive_numpy(self, dtype: np.dtype) -> Tuple[DTypeKind, int, str, str]:
    """
        Build protocol dtype from primitive pandas dtype.

        Parameters
        ----------
        dtype : np.dtype
            Data type to convert from.

        Returns
        -------
        tuple(DTypeKind, bitwidth: int, format_str: str, edianess: str)
        """
    np_kinds = {'i': DTypeKind.INT, 'u': DTypeKind.UINT, 'f': DTypeKind.FLOAT, 'b': DTypeKind.BOOL}
    kind = np_kinds.get(dtype.kind, None)
    if kind is None:
        raise NotImplementedError(f'Data type {dtype} not supported by exchange protocol')
    return (kind, dtype.itemsize * 8, pandas_dtype_to_arrow_c(dtype), dtype.byteorder)