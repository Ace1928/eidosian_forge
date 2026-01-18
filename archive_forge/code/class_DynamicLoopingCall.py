import logging
import sys
from eventlet import event
from eventlet import greenthread
from oslo_utils import timeutils
class DynamicLoopingCall(LoopingCallBase):
    """A looping call which sleeps until the next known event.

    The function called should return how long to sleep for before being
    called again.
    """

    def start(self, initial_delay=None, periodic_interval_max=None):
        self._running = True
        done = event.Event()

        def _inner():
            if initial_delay:
                greenthread.sleep(initial_delay)
            try:
                while self._running:
                    idle = self.f(*self.args, **self.kw)
                    if not self._running:
                        break
                    if periodic_interval_max is not None:
                        idle = min(idle, periodic_interval_max)
                    LOG.debug('Dynamic looping call sleeping for %.02f seconds', idle)
                    greenthread.sleep(idle)
            except LoopingCallDone as e:
                self.stop()
                done.send(e.retvalue)
            except Exception:
                done.send_exception(*sys.exc_info())
                return
            else:
                done.send(True)
        self.done = done
        greenthread.spawn(_inner)
        return self.done