import pickle
import os
import zlib
import inspect
from io import BytesIO
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler
from .numpy_pickle_utils import _ensure_native_byte_order
def _open_pickle(self, file_handle):
    return BytesIO(read_zfile(file_handle))