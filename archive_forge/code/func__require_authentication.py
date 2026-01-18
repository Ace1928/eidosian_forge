import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def _require_authentication(self):
    tcs = self.server.test_case_server
    tcs.auth_required_errors += 1
    self.send_response(tcs.auth_error_code)
    self.send_header_auth_reqed()
    self.send_header('Content-Length', '0')
    self.end_headers()
    return