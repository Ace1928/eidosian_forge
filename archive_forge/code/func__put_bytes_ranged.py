import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _put_bytes_ranged(self, relpath, bytes, at):
    """Append the file-like object part to the end of the location.

        :param relpath: Location to put the contents, relative to base.
        :param bytes:   A string of bytes to upload
        :param at:      The position in the file to add the bytes
        """
    abspath = self._remote_path(relpath)
    headers = {'Accept': '*/*', 'Content-type': 'application/octet-stream', 'Content-Range': 'bytes %d-%d/*' % (at, at + len(bytes) - 1)}
    response = self.request('PUT', abspath, body=bytes, headers=headers)
    code = response.status
    if code in (403, 404, 409):
        raise transport.NoSuchFile(abspath)
    if code not in (200, 201, 204):
        raise self._raise_http_error(abspath, response, 'put file failed')