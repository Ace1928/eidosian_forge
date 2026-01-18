import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _buffered_write_file(fobj):
    """Return a buffered version of a write file object."""
    return io.BufferedWriter(fobj, buffer_size=_IO_BUFFER_SIZE)