import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _is_cid_keyed(self):
    return bool(self.face_flags & FT_FACE_FLAG_CID_KEYED)