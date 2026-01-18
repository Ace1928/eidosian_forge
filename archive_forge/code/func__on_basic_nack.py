import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_basic_nack(self, delivery_tag, multiple):
    for callback in self.events['basic_nack']:
        callback(delivery_tag, multiple)