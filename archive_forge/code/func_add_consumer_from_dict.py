from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from . import messaging
from .entity import Exchange, Queue
def add_consumer_from_dict(self, queue, **options):
    return self.add_queue(Queue.from_dict(queue, **options))