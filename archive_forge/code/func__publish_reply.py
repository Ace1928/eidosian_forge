from __future__ import annotations
import socket
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import copy
from itertools import count
from time import time
from . import Consumer, Exchange, Producer, Queue
from .clocks import LamportClock
from .common import maybe_declare, oid_from
from .exceptions import InconsistencyError
from .log import get_logger
from .matcher import match
from .utils.functional import maybe_evaluate, reprcall
from .utils.objects import cached_property
from .utils.uuid import uuid
def _publish_reply(self, reply, exchange, routing_key, ticket, channel=None, producer=None, **opts):
    chan = channel or self.connection.default_channel
    exchange = Exchange(exchange, exchange_type='direct', delivery_mode='transient', durable=False)
    with self.producer_or_acquire(producer, chan) as producer:
        try:
            producer.publish(reply, exchange=exchange, routing_key=routing_key, declare=[exchange], headers={'ticket': ticket, 'clock': self.clock.forward()}, retry=True, **opts)
        except InconsistencyError:
            pass