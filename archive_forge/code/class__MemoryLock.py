import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
class _MemoryLock:
    """This makes a lock."""

    def __init__(self, path, transport):
        self.path = path
        self.transport = transport
        if self.path in self.transport._locks:
            raise LockError('File {!r} already locked'.format(self.path))
        self.transport._locks[self.path] = self

    def unlock(self):
        del self.transport._locks[self.path]
        self.transport = None