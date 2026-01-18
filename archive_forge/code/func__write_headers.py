import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _write_headers(self, headers):
    """Write the current part's headers to the buffer."""
    return self._write(encode_with(headers, self.encoding))