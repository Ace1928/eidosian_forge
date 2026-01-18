import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
class SynchronizedBase(object):

    def __init__(self, obj, lock=None, ctx=None):
        self._obj = obj
        if lock:
            self._lock = lock
        else:
            ctx = ctx or get_context(force=True)
            self._lock = ctx.RLock()
        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __reduce__(self):
        assert_spawning(self)
        return (synchronized, (self._obj, self._lock))

    def get_obj(self):
        return self._obj

    def get_lock(self):
        return self._lock

    def __repr__(self):
        return '<%s wrapper for %s>' % (type(self).__name__, self._obj)