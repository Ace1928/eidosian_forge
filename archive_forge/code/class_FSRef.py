from __future__ import unicode_literals
from ctypes import cdll, byref, Structure, c_char, c_char_p
from ctypes.util import find_library
from send2trash.compat import binary_type
from send2trash.util import preprocess_paths
class FSRef(Structure):
    _fields_ = [('hidden', c_char * 80)]