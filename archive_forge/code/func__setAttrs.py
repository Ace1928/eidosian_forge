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
def _setAttrs(attrs):
    attrs['atime'] = 0
    d = self.client.setAttrs(b'testfile1', attrs)
    self._emptyBuffers()
    d.addCallback(_getAttrs2)
    d.addCallback(self.assertEqual, attrs)
    return d