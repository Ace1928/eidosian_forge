from __future__ import annotations
import sys
from .compression import decompress
from .exceptions import MessageStateError, reraise
from .serialization import loads
from .utils.functional import dictfilter
@property
def acknowledged(self):
    """Set to true if the message has been acknowledged."""
    return self._state in ACK_STATES