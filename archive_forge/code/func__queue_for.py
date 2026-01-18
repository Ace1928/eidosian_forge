from __future__ import annotations
from collections import defaultdict
from queue import Queue
from . import base, virtual
def _queue_for(self, queue):
    if queue not in self.queues:
        self.queues[queue] = Queue()
    return self.queues[queue]