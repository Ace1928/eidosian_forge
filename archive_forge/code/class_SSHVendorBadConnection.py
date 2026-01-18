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
class SSHVendorBadConnection(TestCaseWithTransport):
    """Test that the ssh vendors handle bad connection properly

    We don't subclass TestCaseWithSFTPServer, because we don't actually
    need an SFTP connection.
    """

    def setUp(self):
        self.requireFeature(features.paramiko)
        super().setUp()
        s = socket.socket()
        s.bind(('localhost', 0))
        self.addCleanup(s.close)
        self.bogus_url = 'sftp://%s:%s/' % s.getsockname()

    def set_vendor(self, vendor, subprocess_stderr=None):
        from breezy.transport import ssh
        self.overrideAttr(ssh._ssh_vendor_manager, '_cached_ssh_vendor', vendor)
        if subprocess_stderr is not None:
            self.overrideAttr(ssh.SubprocessVendor, '_stderr_target', subprocess_stderr)

    def test_bad_connection_paramiko(self):
        """Test that a real connection attempt raises the right error"""
        from breezy.transport import ssh
        self.set_vendor(ssh.ParamikoVendor())
        t = _mod_transport.get_transport_from_url(self.bogus_url)
        self.assertRaises(errors.ConnectionError, t.get, 'foobar')

    def test_bad_connection_ssh(self):
        """None => auto-detect vendor"""
        f = open(os.devnull, 'wb')
        self.addCleanup(f.close)
        self.set_vendor(None, f)
        t = _mod_transport.get_transport_from_url(self.bogus_url)
        try:
            self.assertRaises(errors.ConnectionError, t.get, 'foobar')
        except NameError as e:
            if "global name 'SSHException'" in str(e):
                self.knownFailure('Known NameError bug in paramiko 1.6.1')
            raise