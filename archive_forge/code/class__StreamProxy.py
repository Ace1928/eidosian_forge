from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
class _StreamProxy(object):
    """Small proxy class that enables transparent compression
       detection for the Stream interface (mode 'r|*').
    """

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buf = self.fileobj.read(BLOCKSIZE)

    def read(self, size):
        self.read = self.fileobj.read
        return self.buf

    def getcomptype(self):
        if self.buf.startswith(b'\x1f\x8b\x08'):
            return 'gz'
        elif self.buf[0:3] == b'BZh' and self.buf[4:10] == b'1AY&SY':
            return 'bz2'
        elif self.buf.startswith((b']\x00\x00\x80', b'\xfd7zXZ')):
            return 'xz'
        else:
            return 'tar'

    def close(self):
        self.fileobj.close()