import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class HTTPAuthServer(AuthServer):
    """An HTTP server requiring authentication"""

    def init_http_auth(self):
        self.auth_header_sent = 'WWW-Authenticate'
        self.auth_header_recv = 'Authorization'
        self.auth_error_code = 401