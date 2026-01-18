import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _write_closing_boundary(self):
    """Write the bytes necessary to finish a multipart/form-data body."""
    with reset(self._buffer):
        self._buffer.seek(-2, 2)
        self._buffer.write(b'--\r\n')
    return 2