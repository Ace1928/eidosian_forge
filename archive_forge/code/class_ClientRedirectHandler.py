from __future__ import print_function
import logging
import socket
import sys
from six.moves import BaseHTTPServer
from six.moves import http_client
from six.moves import input
from six.moves import urllib
from oauth2client import _helpers
from oauth2client import client
class ClientRedirectHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """A handler for OAuth 2.0 redirects back to localhost.

    Waits for a single request and parses the query parameters
    into the servers query_params and then stops serving.
    """

    def do_GET(self):
        """Handle a GET request.

        Parses the query parameters and prints a message
        if the flow has completed. Note that we can't detect
        if an error occurred.
        """
        self.send_response(http_client.OK)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        parts = urllib.parse.urlparse(self.path)
        query = _helpers.parse_unique_urlencoded(parts.query)
        self.server.query_params = query
        self.wfile.write(b'<html><head><title>Authentication Status</title></head>')
        self.wfile.write(b'<body><p>The authentication flow has completed.</p>')
        self.wfile.write(b'</body></html>')

    def log_message(self, format, *args):
        """Do not log messages to stdout while running as cmd. line program."""