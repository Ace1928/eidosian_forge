from __future__ import annotations
import string
from queue import Empty
from typing import Any, Dict, Set
import azure.core.exceptions
import azure.servicebus.exceptions
import isodate
from azure.servicebus import (ServiceBusClient, ServiceBusMessage,
from azure.servicebus.management import ServiceBusAdministrationClient
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _get_asb_sender(self, queue: str) -> SendReceive:
    queue_obj = self._queue_cache.get(queue, None)
    if queue_obj is None or queue_obj.sender is None:
        sender = self.queue_service.get_queue_sender(queue, keep_alive=self.uamqp_keep_alive_interval)
        queue_obj = self._add_queue_to_cache(queue, sender=sender)
    return queue_obj