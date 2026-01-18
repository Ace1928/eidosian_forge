import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
class _BaseMixin:
    WIDTH = 80
    HEIGHT = 24

    def _assertBuffer(self, lines):
        receivedLines = self.recvlineClient.__bytes__().splitlines()
        expectedLines = lines + [b''] * (self.HEIGHT - len(lines) - 1)
        self.assertEqual(receivedLines, expectedLines)

    def _trivialTest(self, inputLine, output):
        done = self.recvlineClient.expect(b'done')
        self._testwrite(inputLine)

        def finished(ign):
            self._assertBuffer(output)
        return done.addCallback(finished)