from __future__ import annotations
import os
import socket
from queue import Empty
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.json import dumps, loads
from . import virtual
def _get_queue(self, queue_name):
    queue = self._queues.get(queue_name, None)
    if queue is None:
        queue = Queue(self.client, self._get_path(queue_name))
        self._queues[queue_name] = queue
        len(queue)
    return queue