import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _wait_for_tstate_lock(self, block=True, timeout=-1):
    lock = self._tstate_lock
    if lock is None:
        assert self._is_stopped
        return
    try:
        if lock.acquire(block, timeout):
            lock.release()
            self._stop()
    except:
        if lock.locked():
            lock.release()
            self._stop()
        raise