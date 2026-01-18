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
def _readLink(_):
    d = self.client.readLink(b'testLink')
    self._emptyBuffers()
    testFile = FilePath(os.getcwd()).preauthChild(self.testDir.path)
    testFile = testFile.child('testfile1')
    d.addCallback(self.assertEqual, testFile.path)
    return d