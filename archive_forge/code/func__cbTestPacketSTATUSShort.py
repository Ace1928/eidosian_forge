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
def _cbTestPacketSTATUSShort(self, result):
    """
        Assert that the result is a two-tuple containing empty strings, since
        the STATUS packet had neither a message nor a language.
        """
    self.assertEqual(result[0], b'')
    self.assertEqual(result[1], b'')