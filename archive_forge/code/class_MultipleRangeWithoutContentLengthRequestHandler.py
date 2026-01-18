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
class MultipleRangeWithoutContentLengthRequestHandler(http_server.TestingHTTPRequestHandler):
    """Reply to multiple range requests without content length header."""

    def get_multiple_ranges(self, file, file_size, ranges):
        self.send_response(206)
        self.send_header('Accept-Ranges', 'bytes')
        boundary = '%d' % random.randint(0, 2147483647)
        self.send_header('Content-Type', 'multipart/byteranges; boundary=%s' % boundary)
        self.end_headers()
        for start, end in ranges:
            self.wfile.write(b'--%s\r\n' % boundary.encode('ascii'))
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
            self.end_headers()
            self.send_range_content(file, start, end - start + 1)
        self.wfile.write(b'--%s\r\n' % boundary)