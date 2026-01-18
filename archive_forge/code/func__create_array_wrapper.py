import pickle
import os
import warnings
import io
from pathlib import Path
from .compressor import lz4, LZ4_NOT_INSTALLED_ERROR
from .compressor import _COMPRESSORS, register_compressor, BinaryZlibFile
from .compressor import (ZlibCompressorWrapper, GzipCompressorWrapper,
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_utils import _ensure_native_byte_order
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
from .numpy_pickle_compat import ZNDArrayWrapper  # noqa
from .backports import make_memmap
def _create_array_wrapper(self, array):
    """Create and returns a numpy array wrapper from a numpy array."""
    order = 'F' if array.flags.f_contiguous and (not array.flags.c_contiguous) else 'C'
    allow_mmap = not self.buffered and (not array.dtype.hasobject)
    kwargs = {}
    try:
        self.file_handle.tell()
    except io.UnsupportedOperation:
        kwargs = {'numpy_array_alignment_bytes': None}
    wrapper = NumpyArrayWrapper(type(array), array.shape, order, array.dtype, allow_mmap=allow_mmap, **kwargs)
    return wrapper