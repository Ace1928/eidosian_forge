from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
class _Opener(object):
    """Base class for different locking primitives."""

    def __init__(self, filename, mode, fallback_mode):
        """Create an Opener.

        Args:
            filename: string, The pathname of the file.
            mode: string, The preferred mode to access the file with.
            fallback_mode: string, The mode to use if locking fails.
        """
        self._locked = False
        self._filename = filename
        self._mode = mode
        self._fallback_mode = fallback_mode
        self._fh = None
        self._lock_fd = None

    def is_locked(self):
        """Was the file locked."""
        return self._locked

    def file_handle(self):
        """The file handle to the file. Valid only after opened."""
        return self._fh

    def filename(self):
        """The filename that is being locked."""
        return self._filename

    def open_and_lock(self, timeout, delay):
        """Open the file and lock it.

        Args:
            timeout: float, How long to try to lock for.
            delay: float, How long to wait between retries.
        """
        pass

    def unlock_and_close(self):
        """Unlock and close the file."""
        pass