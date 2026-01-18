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
class ConnectionHandler(urllib.request.BaseHandler):
    """Provides connection-sharing by pre-processing requests.

    urllib.request provides no way to access the HTTPConnection object
    internally used. But we need it in order to achieve
    connection sharing. So, we add it to the request just before
    it is processed, and then we override the do_open method for
    http[s] requests in AbstractHTTPHandler.
    """
    handler_order = 1000

    def __init__(self, report_activity=None, ca_certs=None):
        self._report_activity = report_activity
        self.ca_certs = ca_certs

    def create_connection(self, request, http_connection_class):
        host = request.host
        if not host:
            raise urlutils.InvalidURL(request.get_full_url(), 'no host given.')
        try:
            connection = http_connection_class(host, proxied_host=request.proxied_host, report_activity=self._report_activity, ca_certs=self.ca_certs)
        except http.client.InvalidURL as exception:
            raise urlutils.InvalidURL(request.get_full_url(), extra='nonnumeric port')
        return connection

    def capture_connection(self, request, http_connection_class):
        """Capture or inject the request connection.

        Two cases:
        - the request have no connection: create a new one,

        - the request have a connection: this one have been used
          already, let's capture it, so that we can give it to
          another transport to be reused. We don't do that
          ourselves: the Transport object get the connection from
          a first request and then propagate it, from request to
          request or to cloned transports.
        """
        connection = request.connection
        if connection is None:
            connection = self.create_connection(request, http_connection_class)
            request.connection = connection
        connection.set_debuglevel(DEBUG)
        return request

    def http_request(self, request):
        return self.capture_connection(request, HTTPConnection)

    def https_request(self, request):
        return self.capture_connection(request, HTTPSConnection)