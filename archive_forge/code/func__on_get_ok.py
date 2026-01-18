import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_get_ok(self, delivery_tag, redelivered, exchange, routing_key, message_count, msg):
    msg.channel = self
    msg.delivery_info = {'delivery_tag': delivery_tag, 'redelivered': redelivered, 'exchange': exchange, 'routing_key': routing_key, 'message_count': message_count}
    return msg