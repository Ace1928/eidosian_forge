import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _reset_internal_locks(self, is_alive):
    self._started._at_fork_reinit()
    if is_alive:
        if self._tstate_lock is not None:
            self._tstate_lock._at_fork_reinit()
            self._tstate_lock.acquire()
    else:
        self._is_stopped = True
        self._tstate_lock = None