from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary
class CombinedLock:
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.
    """

    def __init__(self, locks):
        self.locks = tuple(set(locks))

    def acquire(self, blocking=True):
        return all((acquire(lock, blocking=blocking) for lock in self.locks))

    def release(self):
        for lock in self.locks:
            lock.release()

    def __enter__(self):
        for lock in self.locks:
            lock.__enter__()

    def __exit__(self, *args):
        for lock in self.locks:
            lock.__exit__(*args)

    def locked(self):
        return any((lock.locked for lock in self.locks))

    def __repr__(self):
        return f'CombinedLock({list(self.locks)!r})'