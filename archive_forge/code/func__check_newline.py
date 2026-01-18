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
def _check_newline(self, newline):
    if newline is not None and (not isinstance(newline, str)):
        raise TypeError('illegal newline type: %r' % (type(newline),))
    if newline not in (None, '', '\n', '\r', '\r\n'):
        raise ValueError('illegal newline value: %r' % (newline,))