import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _decodeExtra(self):
    extra = self.extra
    unpack = struct.unpack
    while len(extra) >= 4:
        tp, ln = unpack('<HH', extra[:4])
        if ln + 4 > len(extra):
            raise BadZipFile('Corrupt extra field %04x (size=%d)' % (tp, ln))
        if tp == 1:
            data = extra[4:ln + 4]
            try:
                if self.file_size in (18446744073709551615, 4294967295):
                    field = 'File size'
                    self.file_size, = unpack('<Q', data[:8])
                    data = data[8:]
                if self.compress_size == 4294967295:
                    field = 'Compress size'
                    self.compress_size, = unpack('<Q', data[:8])
                    data = data[8:]
                if self.header_offset == 4294967295:
                    field = 'Header offset'
                    self.header_offset, = unpack('<Q', data[:8])
            except struct.error:
                raise BadZipFile(f'Corrupt zip64 extra field. {field} not found.') from None
        extra = extra[ln + 4:]