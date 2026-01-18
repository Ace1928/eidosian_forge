from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _copy_buffer_if_needed(buf: 'pyarrow.Buffer', type_: Optional['pyarrow.DataType'], offset: int, length: int) -> 'pyarrow.Buffer':
    """Copy buffer, if needed."""
    import pyarrow as pa
    if type_ is not None and pa.types.is_boolean(type_):
        buf = _copy_bitpacked_buffer_if_needed(buf, offset, length)
    else:
        type_bytewidth = type_.bit_width // 8 if type_ is not None else 1
        buf = _copy_normal_buffer_if_needed(buf, type_bytewidth, offset, length)
    return buf