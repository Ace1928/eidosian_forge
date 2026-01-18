import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _maintain_shutdown_locks():
    """
    Drop any shutdown locks that don't correspond to running threads anymore.

    Calling this from time to time avoids an ever-growing _shutdown_locks
    set when Thread objects are not joined explicitly. See bpo-37788.

    This must be called with _shutdown_locks_lock acquired.
    """
    to_remove = [lock for lock in _shutdown_locks if not lock.locked()]
    _shutdown_locks.difference_update(to_remove)