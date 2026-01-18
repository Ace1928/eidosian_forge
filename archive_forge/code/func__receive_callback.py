from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from .common import maybe_declare
from .compression import compress
from .connection import is_connection, maybe_channel
from .entity import Exchange, Queue, maybe_delivery_mode
from .exceptions import ContentDisallowed
from .serialization import dumps, prepare_accept_content
from .utils.functional import ChannelPromise, maybe_list
def _receive_callback(self, message):
    accept = self.accept
    on_m, channel, decoded = (self.on_message, self.channel, None)
    try:
        m2p = getattr(channel, 'message_to_python', None)
        if m2p:
            message = m2p(message)
        if accept is not None:
            message.accept = accept
        if message.errors:
            return message._reraise_error(self.on_decode_error)
        decoded = None if on_m else message.decode()
    except Exception as exc:
        if not self.on_decode_error:
            raise
        self.on_decode_error(message, exc)
    else:
        return on_m(message) if on_m else self.receive(decoded, message)