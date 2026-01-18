import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class BasicAuthRequestHandler(AuthRequestHandler):
    """Implements the basic authentication of a request"""

    def authorized(self):
        tcs = self.server.test_case_server
        if tcs.auth_scheme != 'basic':
            return False
        auth_header = self.headers.get(tcs.auth_header_recv, None)
        if auth_header:
            scheme, raw_auth = auth_header.split(' ', 1)
            if scheme.lower() == tcs.auth_scheme:
                user, password = base64.b64decode(raw_auth).split(b':')
                return tcs.authorized(user.decode('ascii'), password.decode('ascii'))
        return False

    def send_header_auth_reqed(self):
        tcs = self.server.test_case_server
        self.send_header(tcs.auth_header_sent, 'Basic realm="%s"' % tcs.auth_realm)