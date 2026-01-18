import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
class MemoryStat:

    def __init__(self, size, kind, perms=None):
        self.st_size = size
        if not S_ISDIR(kind):
            if perms is None:
                perms = 420
            self.st_mode = kind | perms
        else:
            if perms is None:
                perms = 493
            self.st_mode = kind | perms