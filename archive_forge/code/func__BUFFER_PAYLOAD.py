from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _BUFFER_PAYLOAD(self, fragment: Fragment) -> None:
    payload = self._assume_binary(fragment)
    if self._buf_header is None:
        raise ValidationError('Consuming a buffer payload, but current buffer header is None')
    header = BufferHeader(id=self._buf_header['id'])
    cast(Message[Any], self._partial).assemble_buffer(header, payload)
    self._check_complete()