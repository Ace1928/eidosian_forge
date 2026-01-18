import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class ConnectionPoolTests(unittest.TestCase):
    """
    Unit tests for L{ConnectionPool}.
    """

    def test_runWithConnectionRaiseOriginalError(self):
        """
        If rollback fails, L{ConnectionPool.runWithConnection} raises the
        original exception and log the error of the rollback.
        """

        class ConnectionRollbackRaise:

            def __init__(self, pool):
                pass

            def rollback(self):
                raise RuntimeError('problem!')

        def raisingFunction(connection):
            raise ValueError('foo')
        pool = DummyConnectionPool()
        pool.connectionFactory = ConnectionRollbackRaise
        d = pool.runWithConnection(raisingFunction)
        d = self.assertFailure(d, ValueError)

        def cbFailed(ignored):
            errors = self.flushLoggedErrors(RuntimeError)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].value.args[0], 'problem!')
        d.addCallback(cbFailed)
        return d

    def test_closeLogError(self):
        """
        L{ConnectionPool._close} logs exceptions.
        """

        class ConnectionCloseRaise:

            def close(self):
                raise RuntimeError('problem!')
        pool = DummyConnectionPool()
        pool._close(ConnectionCloseRaise())
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.args[0], 'problem!')

    def test_runWithInteractionRaiseOriginalError(self):
        """
        If rollback fails, L{ConnectionPool.runInteraction} raises the
        original exception and log the error of the rollback.
        """

        class ConnectionRollbackRaise:

            def __init__(self, pool):
                pass

            def rollback(self):
                raise RuntimeError('problem!')

        class DummyTransaction:

            def __init__(self, pool, connection):
                pass

        def raisingFunction(transaction):
            raise ValueError('foo')
        pool = DummyConnectionPool()
        pool.connectionFactory = ConnectionRollbackRaise
        pool.transactionFactory = DummyTransaction
        d = pool.runInteraction(raisingFunction)
        d = self.assertFailure(d, ValueError)

        def cbFailed(ignored):
            errors = self.flushLoggedErrors(RuntimeError)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].value.args[0], 'problem!')
        d.addCallback(cbFailed)
        return d

    def test_unstartedClose(self):
        """
        If L{ConnectionPool.close} is called without L{ConnectionPool.start}
        having been called, the pool's startup event is cancelled.
        """
        reactor = EventReactor(False)
        pool = ConnectionPool('twisted.test.test_adbapi', cp_reactor=reactor)
        self.assertEqual(reactor.triggers, [('after', 'startup', pool._start)])
        pool.close()
        self.assertFalse(reactor.triggers)

    def test_startedClose(self):
        """
        If L{ConnectionPool.close} is called after it has been started, but
        not by its shutdown trigger, the shutdown trigger is cancelled.
        """
        reactor = EventReactor(True)
        pool = ConnectionPool('twisted.test.test_adbapi', cp_reactor=reactor)
        self.assertEqual(reactor.triggers, [('during', 'shutdown', pool.finalClose)])
        pool.close()
        self.assertFalse(reactor.triggers)