from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
def _expire_old_connections(self, now):
    """Iterates through the open connections contained in the pool, closing
        ones that have remained idle for longer than max_idle seconds, or have
        been in existence for longer than max_age seconds.

        *now* is the current time, as returned by time.time().
        """
    original_count = len(self.free_items)
    expired = [conn for last_used, created_at, conn in self.free_items if self._is_expired(now, last_used, created_at)]
    new_free = [(last_used, created_at, conn) for last_used, created_at, conn in self.free_items if not self._is_expired(now, last_used, created_at)]
    self.free_items.clear()
    self.free_items.extend(new_free)
    self.current_size -= original_count - len(self.free_items)
    for conn in expired:
        self._safe_close(conn, quiet=True)