import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class ManholeMainTests(unittest.TestCase):
    """
    Test the I{main} method from the I{manhole} module.
    """
    if stdio is None:
        skip = 'Terminal requirements missing'

    def test_mainClassNotFound(self):
        """
        Will raise an exception when called with an argument which is a
        dotted patch which can not be imported..
        """
        exception = self.assertRaises(ValueError, stdio.main, argv=['no-such-class'])
        self.assertEqual('Empty module name', exception.args[0])