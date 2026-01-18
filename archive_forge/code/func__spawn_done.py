import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
def _spawn_done(self, coro):
    self.sem.release()
    if coro is not None:
        self.coroutines_running.remove(coro)
    if self.sem.balance == self.size:
        self.no_coros_running.send(None)