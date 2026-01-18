import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _checkReadable(self):
    if not self._readable:
        raise UnsupportedOperation('File not open for reading')