import collections
import heapq
from . import events
from . import futures
from . import locks
from .tasks import coroutine
def _consume_done_getters(self):
    while self._getters and self._getters[0].done():
        self._getters.popleft()