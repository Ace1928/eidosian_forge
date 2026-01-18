import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def _attempted_range_header(self, offsets, tail_amount):
    """Prepare a HTTP Range header at a level the server should accept.

        :return: the range header representing offsets/tail_amount or None if
            no header can be built.
        """
    if self._range_hint == 'multi':
        return self._range_header(offsets, tail_amount)
    elif self._range_hint == 'single':
        if len(offsets) > 0:
            if tail_amount not in (0, None):
                return None
            else:
                start = offsets[0].start
                last = offsets[-1]
                end = last.start + last.length - 1
                whole = self._coalesce_offsets([(start, end - start + 1)], limit=0, fudge_factor=0)
                return self._range_header(list(whole), 0)
        else:
            return self._range_header(offsets, tail_amount)
    else:
        return None