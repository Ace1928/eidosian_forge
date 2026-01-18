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
class InterProcessLock:
    """An interprocess lock."""
    MAX_DELAY = 0.1
    DELAY_INCREMENT = 0.01

    def __init__(self, path: Union[Path, str], sleep_func: Callable[[float], None]=time.sleep, logger: Optional[logging.Logger]=None):
        """
        args:
            path:
                Path to the file that will be used for locking.
            sleep_func:
                Optional function to use for sleeping.
            logger:
                Optional logger to use for logging.
        """
        self.lockfile = None
        self.path = _utils.canonicalize_path(path)
        self.acquired = False
        self.sleep_func = sleep_func
        self.logger = _utils.pick_first_not_none(logger, LOG)

    def _try_acquire(self, blocking, watch):
        try:
            self.trylock()
        except IOError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                if not blocking or watch.expired():
                    return False
                else:
                    raise _utils.RetryAgain()
            else:
                raise threading.ThreadError('Unable to acquire lock on `%(path)s` due to %(exception)s' % {'path': self.path, 'exception': e})
        else:
            return True

    def _do_open(self):
        basedir = os.path.dirname(self.path)
        if basedir:
            made_basedir = _ensure_tree(basedir)
            if made_basedir:
                self.logger.log(_utils.BLATHER, 'Created lock base path `%s`', basedir)
        if self.lockfile is None or self.lockfile.closed:
            self.lockfile = open(self.path, 'a')

    def acquire(self, blocking: bool=True, delay: float=0.01, max_delay: float=0.1, timeout: Optional[float]=None) -> bool:
        """Attempt to acquire the lock.

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
            gotten = r(self._try_acquire, blocking, watch)
        if not gotten:
            return False
        else:
            self.acquired = True
            self.logger.log(_utils.BLATHER, 'Acquired file lock `%s` after waiting %0.3fs [%s attempts were required]', self.path, watch.elapsed(), r.attempts)
            return True

    def _do_close(self):
        if self.lockfile is not None:
            self.lockfile.close()
            self.lockfile = None

    def __enter__(self):
        gotten = self.acquire()
        if not gotten:
            raise threading.ThreadError('Unable to acquire a file lock on `%s` (when used as a context manager)' % self.path)
        return self

    def release(self):
        """Release the previously acquired lock."""
        if not self.acquired:
            raise threading.ThreadError('Unable to release an unaquired lock')
        try:
            self.unlock()
        except Exception as e:
            msg = ('Could not unlock the acquired lock opened on `%s`', self.path)
            self.logger.exception(msg)
            raise threading.ThreadError(msg) from e
        else:
            self.acquired = False
            try:
                self._do_close()
            except IOError:
                self.logger.exception('Could not close the file handle opened on `%s`', self.path)
            else:
                self.logger.log(_utils.BLATHER, 'Unlocked and closed file lock open on `%s`', self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def exists(self):
        return os.path.exists(self.path)

    def trylock(self):
        _interprocess_mechanism.trylock(self.lockfile)

    def unlock(self):
        _interprocess_mechanism.unlock(self.lockfile)