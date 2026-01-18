import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
class SFTPTestBase(TestCase):

    def setUp(self):
        self.testDir = FilePath(self.mktemp())
        self.testDir = self.testDir.child('extra')
        self.testDir.child('testDirectory').makedirs(True)
        with self.testDir.child('testfile1').open(mode='wb') as f:
            f.write(b'a' * 10 + b'b' * 10)
            with open('/dev/urandom', 'rb') as f2:
                f.write(f2.read(1024 * 64))
        self.testDir.child('testfile1').chmod(420)
        with self.testDir.child('testRemoveFile').open(mode='wb') as f:
            f.write(b'a')
        with self.testDir.child('testRenameFile').open(mode='wb') as f:
            f.write(b'a')
        with self.testDir.child('.testHiddenFile').open(mode='wb') as f:
            f.write(b'a')