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
class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.

    Attributes
    ----------
    mmap_mode: str
        The memorymap mode to use for reading numpy arrays.
    file_handle: file_like
        File object to unpickle from.
    filename: str
        Name of the file to unpickle from. It should correspond to file_handle.
        This parameter is required when using mmap_mode.
    np: module
        Reference to numpy module if numpy is installed else None.

    """
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        self.filename = filename
        self.compat_mode = False
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects, NDArrayWrapper or
        NumpyArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        NDArrayWrapper is used for backward compatibility with joblib <= 0.9.
        """
        Unpickler.load_build(self)
        if isinstance(self.stack[-1], (NDArrayWrapper, NumpyArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, but numpy didn't import correctly")
            array_wrapper = self.stack.pop()
            if isinstance(array_wrapper, NDArrayWrapper):
                self.compat_mode = True
            self.stack.append(array_wrapper.read(self))
    dispatch[pickle.BUILD[0]] = load_build