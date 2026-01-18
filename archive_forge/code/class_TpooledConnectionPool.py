from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
class TpooledConnectionPool(BaseConnectionPool):
    """A pool which gives out :class:`~eventlet.tpool.Proxy`-based database
    connections.
    """

    def create(self):
        now = time.time()
        return (now, now, self.connect(self._db_module, self.connect_timeout, *self._args, **self._kwargs))

    @classmethod
    def connect(cls, db_module, connect_timeout, *args, **kw):
        t = timeout.Timeout(connect_timeout, ConnectTimeout())
        try:
            from eventlet import tpool
            conn = tpool.execute(db_module.connect, *args, **kw)
            return tpool.Proxy(conn, autowrap_names=('cursor',))
        finally:
            t.cancel()