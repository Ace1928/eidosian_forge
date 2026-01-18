from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
class LockedFile(object):
    """Represent a file that has exclusive access."""

    @util.positional(4)
    def __init__(self, filename, mode, fallback_mode, use_native_locking=True):
        """Construct a LockedFile.

        Args:
            filename: string, The path of the file to open.
            mode: string, The mode to try to open the file with.
            fallback_mode: string, The mode to use if locking fails.
            use_native_locking: bool, Whether or not fcntl/win32 locking is
                                used.
        """
        opener = None
        if not opener and use_native_locking:
            try:
                from oauth2client.contrib._win32_opener import _Win32Opener
                opener = _Win32Opener(filename, mode, fallback_mode)
            except ImportError:
                try:
                    from oauth2client.contrib._fcntl_opener import _FcntlOpener
                    opener = _FcntlOpener(filename, mode, fallback_mode)
                except ImportError:
                    pass
        if not opener:
            opener = _PosixOpener(filename, mode, fallback_mode)
        self._opener = opener

    def filename(self):
        """Return the filename we were constructed with."""
        return self._opener._filename

    def file_handle(self):
        """Return the file_handle to the opened file."""
        return self._opener.file_handle()

    def is_locked(self):
        """Return whether we successfully locked the file."""
        return self._opener.is_locked()

    def open_and_lock(self, timeout=0, delay=0.05):
        """Open the file, trying to lock it.

        Args:
            timeout: float, The number of seconds to try to acquire the lock.
            delay: float, The number of seconds to wait between retry attempts.

        Raises:
            AlreadyLockedException: if the lock is already acquired.
            IOError: if the open fails.
        """
        self._opener.open_and_lock(timeout, delay)

    def unlock_and_close(self):
        """Unlock and close a file."""
        self._opener.unlock_and_close()