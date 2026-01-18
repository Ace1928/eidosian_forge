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
class LockedFD:
    """
    This class facilitates a safe read and write operation to a file on disk.
    If we write to 'file', we obtain a lock file at 'file.lock' and write to
    that instead. If we succeed, the lock file will be renamed to overwrite
    the original file.

    When reading, we obtain a lock file, but to prevent other writers from
    succeeding while we are reading the file.

    This type handles error correctly in that it will assure a consistent state
    on destruction.

    **note** with this setup, parallel reading is not possible"""
    __slots__ = ('_filepath', '_fd', '_write')

    def __init__(self, filepath):
        """Initialize an instance with the givne filepath"""
        self._filepath = filepath
        self._fd = None
        self._write = None

    def __del__(self):
        if self._fd is not None:
            self.rollback()

    def _lockfilepath(self):
        return '%s.lock' % self._filepath

    def open(self, write=False, stream=False):
        """
        Open the file descriptor for reading or writing, both in binary mode.

        :param write: if True, the file descriptor will be opened for writing. Other
            wise it will be opened read-only.
        :param stream: if True, the file descriptor will be wrapped into a simple stream
            object which supports only reading or writing
        :return: fd to read from or write to. It is still maintained by this instance
            and must not be closed directly
        :raise IOError: if the lock could not be retrieved
        :raise OSError: If the actual file could not be opened for reading

        **note** must only be called once"""
        if self._write is not None:
            raise AssertionError('Called %s multiple times' % self.open)
        self._write = write
        binary = getattr(os, 'O_BINARY', 0)
        lockmode = os.O_WRONLY | os.O_CREAT | os.O_EXCL | binary
        try:
            fd = os.open(self._lockfilepath(), lockmode, int('600', 8))
            if not write:
                os.close(fd)
            else:
                self._fd = fd
        except OSError as e:
            raise OSError('Lock at %r could not be obtained' % self._lockfilepath()) from e
        if self._fd is None:
            try:
                self._fd = os.open(self._filepath, os.O_RDONLY | binary)
            except:
                remove(self._lockfilepath())
                raise
        if stream:
            from gitdb.stream import FDStream
            return FDStream(self._fd)
        else:
            return self._fd

    def commit(self):
        """When done writing, call this function to commit your changes into the
        actual file.
        The file descriptor will be closed, and the lockfile handled.

        **Note** can be called multiple times"""
        self._end_writing(successful=True)

    def rollback(self):
        """Abort your operation without any changes. The file descriptor will be
        closed, and the lock released.

        **Note** can be called multiple times"""
        self._end_writing(successful=False)

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