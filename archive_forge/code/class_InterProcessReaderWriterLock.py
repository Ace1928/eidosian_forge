from contextlib import contextmanager
import errno
import functools
import logging
import os
from pathlib import Path
import threading
import time
from typing import Callable
from typing import Optional
from typing import Union
from fasteners import _utils
from fasteners.process_mechanism import _interprocess_mechanism
from fasteners.process_mechanism import _interprocess_reader_writer_mechanism
class InterProcessReaderWriterLock:
    """An interprocess readers writer lock."""
    MAX_DELAY = 0.1
    DELAY_INCREMENT = 0.01

    def __init__(self, path: Union[Path, str], sleep_func: Callable[[float], None]=time.sleep, logger: Optional[logging.Logger]=None):
        """
        Args:
            path:
                Path to the file that will be used for locking.
            sleep_func:
                Optional function to use for sleeping.
            logger:
                Optional logger to use for logging.
        """
        self.lockfile = None
        self.path = _utils.canonicalize_path(path)
        self.sleep_func = sleep_func
        self.logger = _utils.pick_first_not_none(logger, LOG)

    @contextmanager
    def read_lock(self, delay=0.01, max_delay=0.1):
        """Context manager that grans a read lock"""
        self.acquire_read_lock(blocking=True, delay=delay, max_delay=max_delay, timeout=None)
        try:
            yield
        finally:
            self.release_read_lock()

    @contextmanager
    def write_lock(self, delay=0.01, max_delay=0.1):
        """Context manager that grans a write lock"""
        gotten = self.acquire_write_lock(blocking=True, delay=delay, max_delay=max_delay, timeout=None)
        if not gotten:
            raise threading.ThreadError('Unable to acquire a file lock on `%s` (when used as a context manager)' % self.path)
        try:
            yield
        finally:
            self.release_write_lock()

    def _try_acquire(self, blocking, watch, exclusive):
        try:
            gotten = _interprocess_reader_writer_mechanism.trylock(self.lockfile, exclusive)
        except Exception as e:
            raise threading.ThreadError('Unable to acquire lock on {} due to {}!'.format(self.path, e))
        if gotten:
            return True
        if not blocking or watch.expired():
            return False
        raise _utils.RetryAgain()

    def _do_open(self):
        basedir = os.path.dirname(self.path)
        if basedir:
            made_basedir = _ensure_tree(basedir)
            if made_basedir:
                self.logger.log(_utils.BLATHER, 'Created lock base path `%s`', basedir)
        if self.lockfile is None:
            self.lockfile = _interprocess_reader_writer_mechanism.get_handle(self.path)

    def acquire_read_lock(self, blocking: bool=True, delay: float=0.01, max_delay: float=0.1, timeout: float=None) -> bool:
        """Attempt to acquire a reader's lock.

        Args:
            blocking:
                Whether to wait to try to acquire the lock.
            delay:
                When `blocking`, starting delay as well as the delay increment
                (in seconds).
            max_delay:
                When `blocking` the maximum delay in between attempts to
                acquire (in seconds).
            timeout:
                When `blocking`, maximal waiting time (in seconds).

        Returns:
            whether or not the acquisition succeeded
        """
        return self._acquire(blocking, delay, max_delay, timeout, exclusive=False)

    def acquire_write_lock(self, blocking: bool=True, delay: float=0.01, max_delay: float=0.1, timeout: float=None) -> bool:
        """Attempt to acquire a writer's lock.

        Args:
            blocking:
                Whether to wait to try to acquire the lock.
            delay:
                When `blocking`, starting delay as well as the delay increment
                (in seconds).
            max_delay:
                When `blocking` the maximum delay in between attempts to
                acquire (in seconds).
            timeout:
                When `blocking`, maximal waiting time (in seconds).

        Returns:
            whether or not the acquisition succeeded
        """
        return self._acquire(blocking, delay, max_delay, timeout, exclusive=True)

    def _acquire(self, blocking=True, delay=0.01, max_delay=0.1, timeout=None, exclusive=True):
        if delay < 0:
            raise ValueError('Delay must be greater than or equal to zero')
        if timeout is not None and timeout < 0:
            raise ValueError('Timeout must be greater than or equal to zero')
        if delay >= max_delay:
            max_delay = delay
        self._do_open()
        watch = _utils.StopWatch(duration=timeout)
        r = _utils.Retry(delay, max_delay, sleep_func=self.sleep_func, watch=watch)
        with watch:
            gotten = r(self._try_acquire, blocking, watch, exclusive)
        if not gotten:
            return False
        else:
            self.logger.log(_utils.BLATHER, 'Acquired file lock `%s` after waiting %0.3fs [%s attempts were required]', self.path, watch.elapsed(), r.attempts)
            return True

    def _do_close(self):
        if self.lockfile is not None:
            _interprocess_reader_writer_mechanism.close_handle(self.lockfile)
            self.lockfile = None

    def release_write_lock(self):
        """Release the writer's lock."""
        try:
            _interprocess_reader_writer_mechanism.unlock(self.lockfile)
        except IOError:
            self.logger.exception('Could not unlock the acquired lock opened on `%s`', self.path)
        else:
            try:
                self._do_close()
            except IOError:
                self.logger.exception('Could not close the file handle opened on `%s`', self.path)
            else:
                self.logger.log(_utils.BLATHER, 'Unlocked and closed file lock open on `%s`', self.path)

    def release_read_lock(self):
        """Release the reader's lock."""
        try:
            _interprocess_reader_writer_mechanism.unlock(self.lockfile)
        except IOError:
            self.logger.exception('Could not unlock the acquired lock opened on `%s`', self.path)
        else:
            try:
                self._do_close()
            except IOError:
                self.logger.exception('Could not close the file handle opened on `%s`', self.path)
            else:
                self.logger.log(_utils.BLATHER, 'Unlocked and closed file lock open on `%s`', self.path)