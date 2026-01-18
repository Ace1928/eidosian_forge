import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class ManholeInterpreterTests(unittest.TestCase):
    """
    Tests for L{manhole.ManholeInterpreter}.
    """

    def test_resetBuffer(self):
        """
        L{ManholeInterpreter.resetBuffer} should empty the input buffer.
        """
        interpreter = manhole.ManholeInterpreter(None)
        interpreter.buffer.extend(['1', '2'])
        interpreter.resetBuffer()
        self.assertFalse(interpreter.buffer)