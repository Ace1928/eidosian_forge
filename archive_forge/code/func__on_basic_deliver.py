import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_basic_deliver(self, consumer_tag, delivery_tag, redelivered, exchange, routing_key, msg):
    msg.channel = self
    msg.delivery_info = {'consumer_tag': consumer_tag, 'delivery_tag': delivery_tag, 'redelivered': redelivered, 'exchange': exchange, 'routing_key': routing_key}
    try:
        fun = self.callbacks[consumer_tag]
    except KeyError:
        AMQP_LOGGER.warning(REJECTED_MESSAGE_WITHOUT_CALLBACK, delivery_tag, consumer_tag, exchange, routing_key)
        self.basic_reject(delivery_tag, requeue=True)
    else:
        fun(msg)