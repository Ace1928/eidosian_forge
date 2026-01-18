import errno
import fcntl
import time
from oauth2client.contrib import locked_file
class _FcntlOpener(locked_file._Opener):
    """Open, lock, and unlock a file using fcntl.lockf."""

    def open_and_lock(self, timeout, delay):
        """Open the file and lock it.

        Args:
            timeout: float, How long to try to lock for.
            delay: float, How long to wait between retries

        Raises:
            AlreadyLockedException: if the lock is already acquired.
            IOError: if the open fails.
            CredentialsFileSymbolicLinkError: if the file is a symbolic
                                              link.
        """
        if self._locked:
            raise locked_file.AlreadyLockedException('File {0} is already locked'.format(self._filename))
        start_time = time.time()
        locked_file.validate_file(self._filename)
        try:
            self._fh = open(self._filename, self._mode)
        except IOError as e:
            if e.errno in (errno.EPERM, errno.EACCES):
                self._fh = open(self._filename, self._fallback_mode)
                return
        while True:
            try:
                fcntl.lockf(self._fh.fileno(), fcntl.LOCK_EX)
                self._locked = True
                return
            except IOError as e:
                if timeout == 0:
                    raise
                if e.errno != errno.EACCES:
                    raise
                if time.time() - start_time >= timeout:
                    locked_file.logger.warn('Could not lock %s in %s seconds', self._filename, timeout)
                    if self._fh:
                        self._fh.close()
                    self._fh = open(self._filename, self._fallback_mode)
                    return
                time.sleep(delay)

    def unlock_and_close(self):
        """Close and unlock the file using the fcntl.lockf primitive."""
        if self._locked:
            fcntl.lockf(self._fh.fileno(), fcntl.LOCK_UN)
        self._locked = False
        if self._fh:
            self._fh.close()