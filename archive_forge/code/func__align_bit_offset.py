from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _align_bit_offset(buf: 'pyarrow.Buffer', bit_offset: int, byte_length: int) -> 'pyarrow.Buffer':
    """Align the bit offset into the buffer with the front of the buffer by shifting
    the buffer and eliminating the offset.
    """
    import pyarrow as pa
    bytes_ = buf.to_pybytes()
    bytes_as_int = int.from_bytes(bytes_, sys.byteorder)
    bytes_as_int >>= bit_offset
    bytes_ = bytes_as_int.to_bytes(byte_length, sys.byteorder)
    return pa.py_buffer(bytes_)