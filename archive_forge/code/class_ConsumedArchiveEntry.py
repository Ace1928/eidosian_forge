from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
class ConsumedArchiveEntry(ArchiveEntry):
    __slots__ = ()

    def get_blocks(self, **kw):
        raise TypeError('the content of this entry has already been read')