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
class _ConnectRequest(Request):

    def __init__(self, request):
        """Constructor

        :param request: the first request sent to the proxied host, already
            processed by the opener (i.e. proxied_host is already set).
        """
        Request.__init__(self, 'CONNECT', request.get_full_url(), connection=request.connection)
        if request.proxied_host is None:
            raise AssertionError()
        self.proxied_host = request.proxied_host

    def get_selector(self):
        return self.proxied_host

    def set_selector(self, selector):
        self.proxied_host = selector
    selector = property(get_selector, set_selector)

    def set_proxy(self, proxy, type):
        """Set the proxy without remembering the proxied host.

        We already know the proxied host by definition, the CONNECT request
        occurs only when the connection goes through a proxy. The usual
        processing (masquerade the request so that the connection is done to
        the proxy while the request is targeted at another host) does not apply
        here. In fact, the connection is already established with proxy and we
        just want to enable the SSL tunneling.
        """
        urllib.request.Request.set_proxy(self, proxy, type)