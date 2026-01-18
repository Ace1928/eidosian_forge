import threading
import warnings
from oslo_utils import importutils
from oslo_utils import timeutils
class EventletEvent(object):
    """A class that provides consistent eventlet/threading Event API.

    This wraps the eventlet.event.Event class to have the same API as
    the standard threading.Event object.
    """

    def __init__(self, *args, **kwargs):
        super(EventletEvent, self).__init__()
        self.clear()

    def clear(self):
        if getattr(self, '_set', True):
            self._set = False
            self._event = _eventlet.event.Event()

    def is_set(self):
        return self._set
    isSet = is_set

    def set(self):
        if not self._set:
            self._set = True
            self._event.send(True)

    def wait(self, timeout=None):
        with timeutils.StopWatch(timeout) as sw:
            while True:
                event = self._event
                with _eventlet.timeout.Timeout(sw.leftover(return_none=True), False):
                    event.wait()
                    if event is not self._event:
                        continue
                return self.is_set()