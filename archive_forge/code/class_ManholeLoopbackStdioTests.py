import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class ManholeLoopbackStdioTests(_StdioMixin, unittest.TestCase, ManholeLoopbackMixin):
    """
    Test manhole loopback over standard IO.
    """
    if stdio is None:
        skip = 'Terminal requirements missing'
    else:
        serverProtocol = stdio.ConsoleManhole