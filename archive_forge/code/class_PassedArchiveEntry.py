from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
class PassedArchiveEntry(ArchiveEntry):
    __slots__ = ()

    def get_blocks(self, **kw):
        raise TypeError("this entry is passed, it's too late to read its content")