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
def _realPath(_):
    d = self.client.realPath(b'testLink')
    self._emptyBuffers()
    testLink = FilePath(os.getcwd()).preauthChild(self.testDir.path)
    testLink = testLink.child('testfile1')
    d.addCallback(self.assertEqual, testLink.path)
    return d