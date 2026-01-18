import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
class SmartPointer(object):
    """Class to hold a non-managed piece of memory"""

    def __init__(self, raw_pointer, destructor):
        self._raw_pointer = raw_pointer
        self._destructor = destructor

    def get(self):
        return self._raw_pointer

    def release(self):
        rp, self._raw_pointer = (self._raw_pointer, None)
        return rp

    def __del__(self):
        try:
            if self._raw_pointer is not None:
                self._destructor(self._raw_pointer)
                self._raw_pointer = None
        except AttributeError:
            pass