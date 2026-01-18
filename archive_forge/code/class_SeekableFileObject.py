import os
from io import BytesIO
import zipfile
import tempfile
import shutil
import enum
import warnings
from ..core import urlopen, get_remote_file
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
class SeekableFileObject:
    """A readonly wrapper file object that add support for seeking, even if
    the wrapped file object does not. The allows us to stream from http and
    still use Pillow.
    """

    def __init__(self, f):
        self.f = f
        self._i = 0
        self._buffer = b''
        self._have_all = False
        self.closed = False

    def read(self, n=None):
        if n is None:
            pass
        else:
            n = int(n)
            if n < 0:
                n = None
        if not self._have_all:
            more = b''
            if n is None:
                more = self.f.read()
                self._have_all = True
            else:
                want_i = self._i + n
                want_more = want_i - len(self._buffer)
                if want_more > 0:
                    more = self.f.read(want_more)
                    if len(more) < want_more:
                        self._have_all = True
            self._buffer += more
        if n is None:
            res = self._buffer[self._i:]
        else:
            res = self._buffer[self._i:self._i + n]
        self._i += len(res)
        return res

    def tell(self):
        return self._i

    def seek(self, i, mode=0):
        i = int(i)
        if mode == 0:
            if i < 0:
                raise ValueError('negative seek value ' + str(i))
            real_i = i
        elif mode == 1:
            real_i = max(0, self._i + i)
        elif mode == 2:
            if not self._have_all:
                self.read()
            real_i = max(0, len(self._buffer) + i)
        else:
            raise ValueError('invalid whence (%s, should be 0, 1 or 2)' % i)
        if real_i <= len(self._buffer):
            pass
        elif not self._have_all:
            assert real_i > self._i
            self.read(real_i - self._i)
        self._i = real_i
        return self._i

    def close(self):
        self.closed = True
        self.f.close()

    def isatty(self):
        return False

    def seekable(self):
        return True