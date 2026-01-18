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
def _checkSeekable(self, msg=None):
    """Internal: raise UnsupportedOperation if file is not seekable
        """
    if not self.seekable():
        raise UnsupportedOperation('File or stream is not seekable.' if msg is None else msg)