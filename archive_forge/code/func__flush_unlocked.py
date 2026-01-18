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
def _flush_unlocked(self):
    if self.closed:
        raise ValueError('flush on closed file')
    while self._write_buf:
        try:
            n = self.raw.write(self._write_buf)
        except BlockingIOError:
            raise RuntimeError('self.raw should implement RawIOBase: it should not raise BlockingIOError')
        if n is None:
            raise BlockingIOError(errno.EAGAIN, 'write could not complete without blocking', 0)
        if n > len(self._write_buf) or n < 0:
            raise OSError('write() returned incorrect number of bytes')
        del self._write_buf[:n]