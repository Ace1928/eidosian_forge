import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _has_multiple_masters(self):
    return bool(self.face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS)