import collections
import threading
import kombu
from kombu import exceptions as kombu_exceptions
from taskflow.engines.worker_based import dispatcher
from taskflow import logging
def _make_queue(self, routing_key, exchange, channel=None):
    """Make a named queue for the given exchange."""
    queue_name = '%s_%s' % (self._exchange_name, routing_key)
    return kombu.Queue(name=queue_name, routing_key=routing_key, durable=False, exchange=exchange, auto_delete=True, channel=channel)