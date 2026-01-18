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
class HTTPAuthHandler(AbstractAuthHandler):
    """Custom http authentication handler.

    Send the authentication preventively to avoid the roundtrip
    associated with the 401 error and keep the revelant info in
    the auth request attribute.
    """
    auth_required_header = 'www-authenticate'
    auth_header = 'Authorization'

    def get_auth(self, request):
        """Get the auth params from the request"""
        return request.auth

    def set_auth(self, request, auth):
        """Set the auth params for the request"""
        request.auth = auth

    def build_password_prompt(self, auth):
        return self._build_password_prompt(auth)

    def build_username_prompt(self, auth):
        return self._build_username_prompt(auth)

    def http_error_401(self, req, fp, code, msg, headers):
        return self.auth_required(req, headers)