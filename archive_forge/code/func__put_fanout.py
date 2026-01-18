from __future__ import annotations
from collections import defaultdict
from queue import Queue
from . import base, virtual
def _put_fanout(self, exchange, message, routing_key=None, **kwargs):
    for queue in self._lookup(exchange, routing_key):
        self._queue_for(queue).put(message)