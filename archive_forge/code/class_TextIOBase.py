import _io
import abc
from _io import (DEFAULT_BUFFER_SIZE, BlockingIOError, UnsupportedOperation,
class TextIOBase(_io._TextIOBase, IOBase):
    __doc__ = _io._TextIOBase.__doc__