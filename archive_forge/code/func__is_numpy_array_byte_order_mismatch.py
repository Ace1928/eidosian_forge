import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _is_numpy_array_byte_order_mismatch(array):
    """Check if numpy array is having byte order mismatch"""
    return sys.byteorder == 'big' and (array.dtype.byteorder == '<' or (array.dtype.byteorder == '|' and array.dtype.fields and all((e[0].byteorder == '<' for e in array.dtype.fields.values())))) or (sys.byteorder == 'little' and (array.dtype.byteorder == '>' or (array.dtype.byteorder == '|' and array.dtype.fields and all((e[0].byteorder == '>' for e in array.dtype.fields.values())))))