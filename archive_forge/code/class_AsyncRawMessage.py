from __future__ import annotations
import base64
from kombu.message import Message
from kombu.utils.encoding import str_to_bytes
class AsyncRawMessage(BaseAsyncMessage):
    """Raw Message."""