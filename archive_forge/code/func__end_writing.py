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
def _end_writing(self, successful=True):
    """Handle the lock according to the write mode """
    if self._write is None:
        raise AssertionError("Cannot end operation if it wasn't started yet")
    if self._fd is None:
        return
    os.close(self._fd)
    self._fd = None
    lockfile = self._lockfilepath()
    if self._write and successful:
        if sys.platform == 'win32':
            if isfile(self._filepath):
                remove(self._filepath)
        os.rename(lockfile, self._filepath)
        chmod(self._filepath, int('644', 8))
    else:
        remove(lockfile)