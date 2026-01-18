import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
class FileSegment:
    """File-like object to pass to requests."""

    def __init__(self, filename, offset, length):
        self.filename = filename
        self.offset = offset
        self.length = length
        self.pos = 0
        self._file = open(filename, 'rb')
        self.seek(0)

    def tell(self):
        return self._file.tell() - self.offset

    def seek(self, offset, whence=0):
        if whence == 0:
            self._file.seek(self.offset + offset, whence)
        elif whence == 1:
            self._file.seek(offset, whence)
        elif whence == 2:
            self._file.seek(self.offset + self.length - offset, 0)

    def read(self, size=-1):
        remaining = self.length - self.pos
        if remaining <= 0:
            return b''
        to_read = remaining if size < 0 else min(size, remaining)
        chunk = self._file.read(to_read)
        self.pos += len(chunk)
        return chunk

    def reset(self):
        self._file.seek(self.offset, 0)