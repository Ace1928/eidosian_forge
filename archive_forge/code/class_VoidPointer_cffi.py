import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
class VoidPointer_cffi(_VoidPointer):
    """Model a newly allocated pointer to void"""

    def __init__(self):
        self._pp = ffi.new('void *[1]')

    def get(self):
        return self._pp[0]

    def address_of(self):
        return self._pp