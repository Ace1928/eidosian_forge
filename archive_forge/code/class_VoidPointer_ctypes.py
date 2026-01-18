import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
class VoidPointer_ctypes(_VoidPointer):
    """Model a newly allocated pointer to void"""

    def __init__(self):
        self._p = c_void_p()

    def get(self):
        return self._p

    def address_of(self):
        return byref(self._p)