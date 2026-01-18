import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class TransactionTests(unittest.TestCase):
    """
    Tests for the L{Transaction} class.
    """

    def test_reopenLogErrorIfReconnect(self):
        """
        If the cursor creation raises an error in L{Transaction.reopen}, it
        reconnects but log the error occurred.
        """

        class ConnectionCursorRaise:
            count = 0

            def reconnect(self):
                pass

            def cursor(self):
                if self.count == 0:
                    self.count += 1
                    raise RuntimeError('problem!')
        pool = FakePool(None)
        transaction = Transaction(pool, ConnectionCursorRaise())
        transaction.reopen()
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.args[0], 'problem!')