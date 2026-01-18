from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import filter, str
from future import utils
import os
import sys
import ssl
import pprint
import socket
from future.backports.urllib import parse as urllib_parse
from future.backports.http.server import (HTTPServer as _HTTPServer,
from future.backports.test import support
class StatsRequestHandler(BaseHTTPRequestHandler):
    """Example HTTP request handler which returns SSL statistics on GET
    requests.
    """
    server_version = 'StatsHTTPS/1.0'

    def do_GET(self, send_body=True):
        """Serve a GET request."""
        sock = self.rfile.raw._sock
        context = sock.context
        stats = {'session_cache': context.session_stats(), 'cipher': sock.cipher(), 'compression': sock.compression()}
        body = pprint.pformat(stats)
        body = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def do_HEAD(self):
        """Serve a HEAD request."""
        self.do_GET(send_body=False)

    def log_request(self, format, *args):
        if support.verbose:
            BaseHTTPRequestHandler.log_request(self, format, *args)