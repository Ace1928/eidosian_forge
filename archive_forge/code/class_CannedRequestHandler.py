import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
class CannedRequestHandler(http_server.TestingHTTPRequestHandler):
    """Request handler for a unique and pre-defined request.

    The only thing we care about here is that a request is emitted by the
    client and we send back a syntactically correct response.

    We expect to receive a *single* request nothing more (and we won't even
    check what request it is, we just read until an empty line).
    """

    def _handle_one_request(self):
        tcs = self.server.test_case_server
        requestline = self.rfile.readline()
        parse_headers(self.rfile)
        if requestline.startswith(b'POST'):
            self.rfile.readline()
        self.wfile.write(tcs.canned_response)