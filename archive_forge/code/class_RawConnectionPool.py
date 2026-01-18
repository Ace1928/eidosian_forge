from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
class RawConnectionPool(BaseConnectionPool):
    """A pool which gives out plain database connections.
    """

    def create(self):
        now = time.time()
        return (now, now, self.connect(self._db_module, self.connect_timeout, *self._args, **self._kwargs))

    @classmethod
    def connect(cls, db_module, connect_timeout, *args, **kw):
        t = timeout.Timeout(connect_timeout, ConnectTimeout())
        try:
            return db_module.connect(*args, **kw)
        finally:
            t.cancel()