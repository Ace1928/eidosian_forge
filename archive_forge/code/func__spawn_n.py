from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
def _spawn_n(seconds, func, args, kwargs):
    hub = hubs.get_hub()
    g = greenlet.greenlet(func, parent=hub.greenlet)
    t = hub.schedule_call_global(seconds, g.switch, *args, **kwargs)
    return (t, g)