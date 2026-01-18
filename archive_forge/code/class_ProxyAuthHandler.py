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
class ProxyAuthHandler(AbstractAuthHandler):
    """Custom proxy authentication handler.

    Send the authentication preventively to avoid the roundtrip
    associated with the 407 error and keep the revelant info in
    the proxy_auth request attribute..
    """
    auth_required_header = 'proxy-authenticate'
    auth_header = 'Proxy-authorization'

    def get_auth(self, request):
        """Get the auth params from the request"""
        return request.proxy_auth

    def set_auth(self, request, auth):
        """Set the auth params for the request"""
        request.proxy_auth = auth

    def build_password_prompt(self, auth):
        prompt = self._build_password_prompt(auth)
        prompt = 'Proxy ' + prompt
        return prompt

    def build_username_prompt(self, auth):
        prompt = self._build_username_prompt(auth)
        prompt = 'Proxy ' + prompt
        return prompt

    def http_error_407(self, req, fp, code, msg, headers):
        return self.auth_required(req, headers)