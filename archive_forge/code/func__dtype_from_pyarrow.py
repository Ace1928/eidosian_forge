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
def _dtype_from_pyarrow(self, dtype):
    """
        Build protocol dtype from PyArrow type.

        Parameters
        ----------
        dtype : pyarrow.DataType
            Data type to convert from.

        Returns
        -------
        tuple(DTypeKind, bitwidth: int, format_str: str, edianess: str)
        """
    kind = None
    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype) or pa.types.is_time(dtype):
        kind = DTypeKind.DATETIME
        bit_width = dtype.bit_width
    elif pa.types.is_dictionary(dtype):
        kind = DTypeKind.CATEGORICAL
        bit_width = dtype.bit_width
    elif pa.types.is_string(dtype):
        kind = DTypeKind.STRING
        bit_width = 8
    elif pa.types.is_boolean(dtype):
        kind = DTypeKind.BOOL
        bit_width = dtype.bit_width
    if kind is not None:
        return (kind, bit_width, arrow_dtype_to_arrow_c(dtype), Endianness.NATIVE)
    else:
        return self._dtype_from_primitive_numpy(np.dtype(dtype.to_pandas_dtype()))