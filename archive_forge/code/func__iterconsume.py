from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from . import messaging
from .entity import Exchange, Queue
def _iterconsume(connection, consumer, no_ack=False, limit=None):
    consumer.consume(no_ack=no_ack)
    for iteration in count(0):
        if limit and iteration >= limit:
            break
        yield connection.drain_events()