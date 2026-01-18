from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
class PooledConnectionWrapper(GenericConnectionWrapper):
    """A connection wrapper where:
    - the close method returns the connection to the pool instead of closing it directly
    - ``bool(conn)`` returns a reasonable value
    - returns itself to the pool if it gets garbage collected
    """

    def __init__(self, baseconn, pool):
        super().__init__(baseconn)
        self._pool = pool

    def __nonzero__(self):
        return hasattr(self, '_base') and bool(self._base)
    __bool__ = __nonzero__

    def _destroy(self):
        self._pool = None
        try:
            del self._base
        except AttributeError:
            pass

    def close(self):
        """Return the connection to the pool, and remove the
        reference to it so that you can't use it again through this
        wrapper object.
        """
        if self and self._pool:
            self._pool.put(self)
        self._destroy()

    def __del__(self):
        return