from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
def _unwrap_connection(self, conn):
    """If the connection was wrapped by a subclass of
        BaseConnectionWrapper and is still functional (as determined
        by the __nonzero__, or __bool__ in python3, method), returns
        the unwrapped connection.  If anything goes wrong with this
        process, returns None.
        """
    base = None
    try:
        if conn:
            base = conn._base
            conn._destroy()
        else:
            base = None
    except AttributeError:
        pass
    return base