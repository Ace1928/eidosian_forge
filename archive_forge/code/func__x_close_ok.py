import logging
import socket
import uuid
import warnings
from array import array
from time import monotonic
from vine import ensure_promise
from . import __version__, sasl, spec
from .abstract_channel import AbstractChannel
from .channel import Channel
from .exceptions import (AMQPDeprecationWarning, ChannelError, ConnectionError,
from .method_framing import frame_handler, frame_writer
from .transport import Transport
def _x_close_ok(self):
    """Confirm a connection close.

        This method confirms a Connection.Close method and tells the
        recipient that it is safe to release resources for the
        connection and close the socket.

        RULE:
            A peer that detects a socket closure without having
            received a Close-Ok handshake method SHOULD log the error.
        """
    self.send_method(spec.Connection.CloseOk, callback=self._on_close_ok)