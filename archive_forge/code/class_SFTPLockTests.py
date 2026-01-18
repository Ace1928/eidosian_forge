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
class SFTPLockTests(TestCaseWithSFTPServer):

    def test_sftp_locks(self):
        from breezy.errors import LockError
        t = self.get_transport()
        l = t.lock_write('bogus')
        self.assertPathExists('bogus.write-lock')
        self.assertRaises(LockError, t.lock_write, 'bogus')
        l.unlock()
        self.assertFalse(lexists('bogus.write-lock'))
        with open('something.write-lock', 'wb') as f:
            f.write(b'fake lock\n')
        self.assertRaises(LockError, t.lock_write, 'something')
        os.remove('something.write-lock')
        l = t.lock_write('something')
        l2 = t.lock_write('bogus')
        l.unlock()
        l2.unlock()