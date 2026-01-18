from __future__ import annotations
import sys
from .compression import decompress
from .exceptions import MessageStateError, reraise
from .serialization import loads
from .utils.functional import dictfilter
def _reraise_error(self, callback=None):
    try:
        reraise(*self.errors[0])
    except Exception as exc:
        if not callback:
            raise
        callback(self, exc)