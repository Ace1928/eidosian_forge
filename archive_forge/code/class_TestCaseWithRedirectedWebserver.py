import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class TestCaseWithRedirectedWebserver(TestCaseWithTwoWebservers):
    """A support class providing redirections from one server to another.

    We set up two webservers to allows various tests involving
    redirections.
    The 'old' server is redirected to the 'new' server.
    """

    def setUp(self):
        super().setUp()
        self.new_server = self.get_readonly_server()
        self.old_server = self.get_secondary_server()

    def create_transport_secondary_server(self):
        """Create the secondary server redirecting to the primary server"""
        new = self.get_readonly_server()
        redirecting = HTTPServerRedirecting(protocol_version=self._protocol_version)
        redirecting.redirect_to(new.host, new.port)
        redirecting._url_protocol = self._url_protocol
        return redirecting

    def get_old_url(self, relpath=None):
        base = self.old_server.get_url()
        return self._adjust_url(base, relpath)

    def get_old_transport(self, relpath=None):
        t = transport.get_transport_from_url(self.get_old_url(relpath))
        self.assertTrue(t.is_readonly())
        return t

    def get_new_url(self, relpath=None):
        base = self.new_server.get_url()
        return self._adjust_url(base, relpath)

    def get_new_transport(self, relpath=None):
        t = transport.get_transport_from_url(self.get_new_url(relpath))
        self.assertTrue(t.is_readonly())
        return t