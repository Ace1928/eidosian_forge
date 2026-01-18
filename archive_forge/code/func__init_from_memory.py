import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _init_from_memory(self, library, face, index, byte_stream):
    error = FT_New_Memory_Face(library, byte_stream, len(byte_stream), index, byref(face))
    self._filebodys.append(byte_stream)
    return error