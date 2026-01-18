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
class SFTPTransportTestRelative(TestCaseWithSFTPServer):
    """Test the SFTP transport with homedir based relative paths."""

    def test__remote_path(self):
        if sys.platform == 'darwin':
            self.knownFailure('Mac OSX symlinks /tmp to /private/tmp, testing against self.test_dir is not appropriate')
        t = self.get_transport()
        test_dir = self.test_dir
        if sys.platform == 'win32':
            test_dir = '/' + test_dir
        self.assertIsSameRealPath(test_dir + '/relative', t._remote_path('relative'))
        root_segments = test_dir.split('/')
        root_parent = '/'.join(root_segments[:-1])
        self.assertIsSameRealPath(root_parent + '/sibling', t._remote_path('../sibling'))