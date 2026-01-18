import sys
import heapq
import collections
import traceback
from eventlet.event import Event
from eventlet.greenthread import getcurrent
from eventlet.hubs import get_hub
import queue as Stdlib_Queue
from eventlet.timeout import Timeout
def _schedule_unlock(self):
    if self._event_unlock is None:
        self._event_unlock = get_hub().schedule_call_global(0, self._unlock)