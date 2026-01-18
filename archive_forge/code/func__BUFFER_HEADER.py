from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _BUFFER_HEADER(self, fragment: Fragment) -> None:
    header = json.loads(self._assume_text(fragment))
    if set(header) != {'id'}:
        raise ValidationError(f'Malformed buffer header {header!r}')
    self._buf_header = header
    self._current_consumer = self._BUFFER_PAYLOAD