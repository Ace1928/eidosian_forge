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
def _set_decoded_chars(self, chars):
    """Set the _decoded_chars buffer."""
    self._decoded_chars = chars
    self._decoded_chars_used = 0