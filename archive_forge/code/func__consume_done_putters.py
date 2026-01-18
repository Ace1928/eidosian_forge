import collections
import heapq
from . import events
from . import futures
from . import locks
from .tasks import coroutine
def _consume_done_putters(self):
    while self._putters and self._putters[0][1].done():
        self._putters.popleft()