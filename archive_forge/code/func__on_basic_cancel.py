import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_basic_cancel(self, consumer_tag):
    """Consumer cancelled by server.

        Most likely the queue was deleted.

        """
    callback = self._remove_tag(consumer_tag)
    if callback:
        callback(consumer_tag)
    else:
        raise ConsumerCancelled(consumer_tag, spec.Basic.Cancel)