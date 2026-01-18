import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def _checked_read(self, size):
    """Read the file checking for short reads.

        The data read is discarded along the way.
        """
    pos = self._pos
    remaining = size
    while remaining > 0:
        data = self._file.read(min(remaining, self._discarded_buf_size))
        remaining -= len(data)
        if not data:
            raise errors.ShortReadvError(self._path, pos, size, size - remaining)
    self._pos += size