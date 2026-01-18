import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def _handle_one_request(self):
    tcs = self.server.test_case_server
    requestline = self.rfile.readline()
    parse_headers(self.rfile)
    if requestline.startswith(b'POST'):
        self.rfile.readline()
    self.wfile.write(tcs.canned_response)