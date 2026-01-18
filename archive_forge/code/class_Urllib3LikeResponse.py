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
class Urllib3LikeResponse:

    def __init__(self, actual):
        self._actual = actual
        self._data = None

    def getheader(self, name, default=None):
        if self._actual.headers is None:
            raise http.client.ResponseNotReady()
        return self._actual.headers.get(name, default)

    def getheaders(self):
        if self._actual.headers is None:
            raise http.client.ResponseNotReady()
        return list(self._actual.headers.items())

    @property
    def status(self):
        return self._actual.code

    @property
    def reason(self):
        return self._actual.reason

    @property
    def data(self):
        if self._data is None:
            self._data = self._actual.read()
        return self._data

    @property
    def text(self):
        if self.status == 204:
            return None
        charset = cgi.parse_header(self._actual.headers['Content-Type'])[1].get('charset')
        if charset:
            return self.data.decode(charset)
        else:
            return self.data.decode()

    def read(self, amt=None):
        if amt is None and 'evil' in debug.debug_flags:
            mutter_callsite(4, 'reading full response.')
        return self._actual.read(amt)

    def readlines(self):
        return self._actual.readlines()

    def readline(self, size=-1):
        return self._actual.readline(size)