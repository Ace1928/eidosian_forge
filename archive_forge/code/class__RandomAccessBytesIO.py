import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
class _RandomAccessBytesIO:
    """Wrapper to provide required functionality in case memory maps cannot or may
    not be used. This is only really required in python 2.4"""
    __slots__ = '_sio'

    def __init__(self, buf=''):
        self._sio = BytesIO(buf)

    def __getattr__(self, attr):
        return getattr(self._sio, attr)

    def __len__(self):
        return len(self.getvalue())

    def __getitem__(self, i):
        return self.getvalue()[i]

    def __getslice__(self, start, end):
        return self.getvalue()[start:end]