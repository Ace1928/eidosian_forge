import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_close_ok(self):
    """Confirm a channel close.

        This method confirms a Channel.Close method and tells the
        recipient that it is safe to release resources for the channel
        and close the socket.

        RULE:

            A peer that detects a socket closure without having
            received a Channel.Close-Ok handshake method SHOULD log
            the error.
        """
    self.collect()