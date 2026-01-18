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
def _testRenamed(_, attrs):
    d = self.client.getAttrs(b'testRenamedFile')
    self._emptyBuffers()
    d.addCallback(self.assertEqual, attrs)