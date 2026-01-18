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
class BasicAuthHandler(AbstractAuthHandler):
    """A custom basic authentication handler."""
    scheme = 'basic'
    handler_order = 500
    auth_regexp = re.compile('realm="([^"]*)"', re.I)

    def build_auth_header(self, auth, request):
        raw = '{}:{}'.format(auth['user'], auth['password'])
        auth_header = 'Basic ' + base64.b64encode(raw.encode('utf-8')).decode('ascii')
        return auth_header

    def extract_realm(self, header_value):
        match = self.auth_regexp.search(header_value)
        realm = None
        if match:
            realm = match.group(1)
        return (match, realm)

    def auth_match(self, header, auth):
        scheme, raw_auth = self._parse_auth_header(header)
        if scheme != self.scheme:
            return False
        match, realm = self.extract_realm(raw_auth)
        if match:
            self.update_auth(auth, 'scheme', scheme)
            self.update_auth(auth, 'realm', realm)
            if auth.get('user', None) is None or auth.get('password', None) is None:
                user, password = self.get_user_password(auth)
                self.update_auth(auth, 'user', user)
                self.update_auth(auth, 'password', password)
        return match is not None

    def auth_params_reusable(self, auth):
        return auth.get('scheme', None) == 'basic'