import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
class TqdmDefaultWriteLock(object):
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    You must initialise a `tqdm` or `TqdmDefaultWriteLock` instance
    before forking in order for the write lock to work.
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """
    th_lock = TRLock()

    def __init__(self):
        cls = type(self)
        root_lock = cls.th_lock
        if root_lock is not None:
            root_lock.acquire()
        cls.create_mp_lock()
        self.locks = [lk for lk in [cls.mp_lock, cls.th_lock] if lk is not None]
        if root_lock is not None:
            root_lock.release()

    def acquire(self, *a, **k):
        for lock in self.locks:
            lock.acquire(*a, **k)

    def release(self):
        for lock in self.locks[::-1]:
            lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()

    @classmethod
    def create_mp_lock(cls):
        if not hasattr(cls, 'mp_lock'):
            try:
                from multiprocessing import RLock
                cls.mp_lock = RLock()
            except (ImportError, OSError):
                cls.mp_lock = None

    @classmethod
    def create_th_lock(cls):
        assert hasattr(cls, 'th_lock')
        warn('create_th_lock not needed anymore', TqdmDeprecationWarning, stacklevel=2)