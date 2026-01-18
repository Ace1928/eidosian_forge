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
def _get_buffer_size(self, bit_width: int, is_offset_buffer: bool=False) -> int:
    """
        Compute buffer's size in bytes for the current chunk.

        Parameters
        ----------
        bit_width : int
            Bit width of the underlying data type.
        is_offset_buffer : bool, default: False
            Whether the buffer describes offsets.

        Returns
        -------
        int
            Number of bytes to read from the start of the buffer + offset to retrieve the whole chunk.
        """
    elements_in_buffer = self.size() + 1 if is_offset_buffer else self.size()
    result = ceil(bit_width * elements_in_buffer / 8)
    if bit_width == 1 and self.offset % 8 + self.size() > result * 8:
        result += 1
    return result