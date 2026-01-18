import pickle
import os
import zlib
import inspect
from io import BytesIO
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler
from .numpy_pickle_utils import _ensure_native_byte_order
class ZipNumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles."""
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        """Constructor."""
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _open_pickle(self, file_handle):
        return BytesIO(read_zfile(file_handle))

    def load_build(self):
        """Set the state of a newly created object.

        We capture it to replace our place-holder objects,
        NDArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        """
        Unpickler.load_build(self)
        if isinstance(self.stack[-1], NDArrayWrapper):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, but numpy didn't import correctly")
            nd_array_wrapper = self.stack.pop()
            array = nd_array_wrapper.read(self)
            self.stack.append(array)
    dispatch[pickle.BUILD[0]] = load_build