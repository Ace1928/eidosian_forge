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
def _coalesce_readv(self, relpath, coalesced):
    """Issue several GET requests to satisfy the coalesced offsets"""

    def get_and_yield(relpath, coalesced):
        if coalesced:
            code, rfile = self._get(relpath, coalesced)
            for coal in coalesced:
                yield (coal, rfile)
    if self._range_hint is None:
        yield from get_and_yield(relpath, coalesced)
    else:
        total = len(coalesced)
        if self._range_hint == 'multi':
            max_ranges = self._max_get_ranges
        elif self._range_hint == 'single':
            max_ranges = total
        else:
            raise AssertionError('Unknown _range_hint %r' % (self._range_hint,))
        cumul = 0
        ranges = []
        for coal in coalesced:
            if self._get_max_size > 0 and cumul + coal.length > self._get_max_size or len(ranges) >= max_ranges:
                yield from get_and_yield(relpath, ranges)
                ranges = [coal]
                cumul = coal.length
            else:
                ranges.append(coal)
                cumul += coal.length
        yield from get_and_yield(relpath, ranges)