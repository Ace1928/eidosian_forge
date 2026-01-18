import sys
import heapq
import collections
import traceback
from eventlet.event import Event
from eventlet.greenthread import getcurrent
from eventlet.hubs import get_hub
import queue as Stdlib_Queue
from eventlet.timeout import Timeout
def _put_bookkeeping(self):
    self.unfinished_tasks += 1
    if self._cond.ready():
        self._cond.reset()