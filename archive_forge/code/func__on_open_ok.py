import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_open_ok(self):
    """Signal that the channel is ready.

        This method signals to the client that the channel is ready
        for use.
        """
    self.is_open = True
    self.on_open(self)
    AMQP_LOGGER.debug('Channel open')