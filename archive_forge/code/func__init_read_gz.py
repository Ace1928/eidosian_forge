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
def _init_read_gz(self):
    """Initialize for reading a gzip compressed fileobj.
        """
    self.cmp = self.zlib.decompressobj(-self.zlib.MAX_WBITS)
    self.dbuf = b''
    if self.__read(2) != b'\x1f\x8b':
        raise ReadError('not a gzip file')
    if self.__read(1) != b'\x08':
        raise CompressionError('unsupported compression method')
    flag = ord(self.__read(1))
    self.__read(6)
    if flag & 4:
        xlen = ord(self.__read(1)) + 256 * ord(self.__read(1))
        self.read(xlen)
    if flag & 8:
        while True:
            s = self.__read(1)
            if not s or s == NUL:
                break
    if flag & 16:
        while True:
            s = self.__read(1)
            if not s or s == NUL:
                break
    if flag & 2:
        self.__read(2)