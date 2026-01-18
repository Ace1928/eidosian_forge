from __future__ import annotations
import string
from queue import Empty
from typing import Any, Optional
from azure.core.exceptions import ResourceExistsError
from kombu.utils.encoding import safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _ensure_queue(self, queue):
    """Ensure a queue exists."""
    queue = self.entity_name(self.queue_name_prefix + queue)
    try:
        q = self._queue_service.get_queue_client(queue=self._queue_name_cache[queue])
    except KeyError:
        try:
            q = self.queue_service.create_queue(queue)
        except ResourceExistsError:
            q = self._queue_service.get_queue_client(queue=queue)
        self._queue_name_cache[queue] = q.get_queue_properties()
    return q