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
class TestCaseWithSFTPServer(TestCaseWithTransport):
    """A test case base class that provides a sftp server on localhost."""

    def setUp(self):
        super().setUp()
        self.requireFeature(features.paramiko)
        set_test_transport_to_sftp(self)