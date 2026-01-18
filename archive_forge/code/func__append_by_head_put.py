import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _append_by_head_put(self, relpath, bytes):
    """Append without getting the whole file.

        When the server allows it, a 'Content-Range' header can be specified.
        """
    response = self._head(relpath)
    code = response.status
    if code == 404:
        relpath_size = 0
    else:
        relpath_size = int(response.getheader('Content-Length', 0))
        if relpath_size == 0:
            trace.mutter('if %s is not empty, the server is buggy' % relpath)
    if relpath_size:
        self._put_bytes_ranged(relpath, bytes, relpath_size)
    else:
        self.put_bytes(relpath, bytes)
    return relpath_size