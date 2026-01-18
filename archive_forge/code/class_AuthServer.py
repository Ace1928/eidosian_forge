import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class AuthServer(http_server.HttpServer):
    """Extends HttpServer with a dictionary of passwords.

    This is used as a base class for various schemes which should
    all use or redefined the associated AuthRequestHandler.

    Note that no users are defined by default, so add_user should
    be called before issuing the first request.
    """
    auth_header_sent = None
    auth_header_recv = None
    auth_error_code = None
    auth_realm = 'Thou should not pass'

    def __init__(self, request_handler, auth_scheme, protocol_version=None):
        http_server.HttpServer.__init__(self, request_handler, protocol_version=protocol_version)
        self.auth_scheme = auth_scheme
        self.password_of = {}
        self.auth_required_errors = 0

    def add_user(self, user, password):
        """Declare a user with an associated password.

        password can be empty, use an empty string ('') in that
        case, not None.
        """
        self.password_of[user] = password

    def authorized(self, user, password):
        """Check that the given user provided the right password"""
        expected_password = self.password_of.get(user, None)
        return expected_password is not None and password == expected_password