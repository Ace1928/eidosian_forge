import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
def _spawn_n_impl(self, func, args, kwargs, coro):
    try:
        try:
            func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit, greenlet.GreenletExit):
            raise
        except:
            if DEBUG:
                traceback.print_exc()
    finally:
        if coro is None:
            return
        else:
            coro = eventlet.getcurrent()
            self._spawn_done(coro)