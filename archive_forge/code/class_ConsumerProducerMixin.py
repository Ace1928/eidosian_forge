from __future__ import annotations
import socket
from contextlib import contextmanager
from functools import partial
from itertools import count
from time import sleep
from .common import ignore_errors
from .log import get_logger
from .messaging import Consumer, Producer
from .utils.compat import nested
from .utils.encoding import safe_repr
from .utils.limits import TokenBucket
from .utils.objects import cached_property
class ConsumerProducerMixin(ConsumerMixin):
    """Consumer and Producer mixin.

    Version of ConsumerMixin having separate connection for also
    publishing messages.

    Example:
    -------
        .. code-block:: python

            class Worker(ConsumerProducerMixin):

                def __init__(self, connection):
                    self.connection = connection

                def get_consumers(self, Consumer, channel):
                    return [Consumer(queues=Queue('foo'),
                                     on_message=self.handle_message,
                                     accept='application/json',
                                     prefetch_count=10)]

                def handle_message(self, message):
                    self.producer.publish(
                        {'message': 'hello to you'},
                        exchange='',
                        routing_key=message.properties['reply_to'],
                        correlation_id=message.properties['correlation_id'],
                        retry=True,
                    )
    """
    _producer_connection = None

    def on_consume_end(self, connection, channel):
        if self._producer_connection is not None:
            self._producer_connection.close()
            self._producer_connection = None

    @property
    def producer(self):
        return Producer(self.producer_connection)

    @property
    def producer_connection(self):
        if self._producer_connection is None:
            conn = self.connection.clone()
            conn.ensure_connection(self.on_connection_error, self.connect_max_retries)
            self._producer_connection = conn
        return self._producer_connection