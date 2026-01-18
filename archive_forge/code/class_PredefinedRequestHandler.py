import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
class PredefinedRequestHandler(http_server.TestingHTTPRequestHandler):
    """Request handler for a unique and pre-defined request.

    The only thing we care about here is how many bytes travel on the wire. But
    since we want to measure it for a real http client, we have to send it
    correct responses.

    We expect to receive a *single* request nothing more (and we won't even
    check what request it is, we just measure the bytes read until an empty
    line.
    """

    def _handle_one_request(self):
        tcs = self.server.test_case_server
        requestline = self.rfile.readline()
        headers = parse_headers(self.rfile)
        bytes_read = len(headers.as_bytes())
        bytes_read += headers.as_bytes().count(b'\n')
        bytes_read += len(requestline)
        if requestline.startswith(b'POST'):
            body = self.rfile.readline()
            bytes_read += len(body)
        tcs.bytes_read = bytes_read
        tcs.bytes_written = len(tcs.canned_response)
        self.wfile.write(tcs.canned_response)