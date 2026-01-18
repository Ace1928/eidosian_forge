import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def _seek_to_next_range(self):
    if self._boundary is None:
        raise errors.InvalidRange(self._path, self._pos, 'Range (%s, %s) exhausted' % (self._start, self._size))
    self.read_boundary()
    self.read_range_definition()