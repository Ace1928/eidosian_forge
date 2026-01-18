import logging
import time
from monotonic import monotonic as now  # noqa
class LockStack(object):
    """Simple lock stack to get and release many locks.

    An instance of this should **not** be used by many threads at the
    same time, as the stack that is maintained will be corrupted and
    invalid if that is attempted.
    """

    def __init__(self, logger=None):
        self._stack = []
        self._logger = pick_first_not_none(logger, LOG)

    def acquire_lock(self, lock):
        gotten = lock.acquire()
        if gotten:
            self._stack.append(lock)
        return gotten

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        am_left = len(self._stack)
        tot_am = am_left
        while self._stack:
            lock = self._stack.pop()
            try:
                lock.release()
            except Exception:
                self._logger.exception('Failed releasing lock %s from lock stack with %s locks', am_left, tot_am)
            am_left -= 1