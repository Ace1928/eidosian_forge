import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _ensure_native_byte_order(array):
    """Use the byte order of the host while preserving values

    Does nothing if array already uses the system byte order.
    """
    if _is_numpy_array_byte_order_mismatch(array):
        array = array.byteswap().newbyteorder('=')
    return array