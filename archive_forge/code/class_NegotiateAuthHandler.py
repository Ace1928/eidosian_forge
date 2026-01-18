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
class NegotiateAuthHandler(AbstractAuthHandler):
    """A authentication handler that handles WWW-Authenticate: Negotiate.

    At the moment this handler supports just Kerberos. In the future,
    NTLM support may also be added.
    """
    scheme = 'negotiate'
    handler_order = 480
    requires_username = False

    def auth_match(self, header, auth):
        scheme, raw_auth = self._parse_auth_header(header)
        if scheme != self.scheme:
            return False
        self.update_auth(auth, 'scheme', scheme)
        resp = self._auth_match_kerberos(auth)
        if resp is None:
            return False
        self.update_auth(auth, 'negotiate_response', resp)
        return True

    def _auth_match_kerberos(self, auth):
        """Try to create a GSSAPI response for authenticating against a host."""
        global kerberos, checked_kerberos
        if kerberos is None and (not checked_kerberos):
            try:
                import kerberos
            except ModuleNotFoundError:
                kerberos = None
            checked_kerberos = True
        if kerberos is None:
            return None
        ret, vc = kerberos.authGSSClientInit('HTTP@%(host)s' % auth)
        if ret < 1:
            trace.warning('Unable to create GSSAPI context for %s: %d', auth['host'], ret)
            return None
        ret = kerberos.authGSSClientStep(vc, '')
        if ret < 0:
            trace.mutter('authGSSClientStep failed: %d', ret)
            return None
        return kerberos.authGSSClientResponse(vc)

    def build_auth_header(self, auth, request):
        return 'Negotiate %s' % auth['negotiate_response']

    def auth_params_reusable(self, auth):
        return auth.get('scheme', None) == 'negotiate' and auth.get('negotiate_response', None) is not None