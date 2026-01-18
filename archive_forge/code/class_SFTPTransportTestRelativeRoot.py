import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class SFTPTransportTestRelativeRoot(TestCaseWithSFTPServer):
    """Test the SFTP transport with homedir based relative paths."""

    def setUp(self):
        self._get_remote_is_absolute = False
        super().setUp()

    def test__remote_path_relative_root(self):
        t = self.get_transport('')
        self.assertEqual('/~/', t._parsed_url.path)
        self.assertEqual('a', t._remote_path('a'))