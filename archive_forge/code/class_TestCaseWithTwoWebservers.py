import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class TestCaseWithTwoWebservers(TestCaseWithWebserver):
    """A support class providing readonly urls on two servers that are http://.

    We set up two webservers to allows various tests involving
    proxies or redirections from one server to the other.
    """

    def setUp(self):
        super().setUp()
        self.transport_secondary_server = http_server.HttpServer
        self.__secondary_server = None

    def create_transport_secondary_server(self):
        """Create a transport server from class defined at init.

        This is mostly a hook for daughter classes.
        """
        server = self.transport_secondary_server(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server

    def get_secondary_server(self):
        """Get the server instance for the secondary transport."""
        if self.__secondary_server is None:
            self.__secondary_server = self.create_transport_secondary_server()
            self.start_server(self.__secondary_server)
        return self.__secondary_server

    def get_secondary_url(self, relpath=None):
        base = self.get_secondary_server().get_url()
        return self._adjust_url(base, relpath)

    def get_secondary_transport(self, relpath=None):
        t = transport.get_transport_from_url(self.get_secondary_url(relpath))
        self.assertTrue(t.is_readonly())
        return t