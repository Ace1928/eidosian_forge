import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class DTPFactoryTests(TestCase):
    """
    Tests for L{ftp.DTPFactory}.
    """

    def setUp(self):
        """
        Create a fake protocol interpreter and a L{ftp.DTPFactory} instance to
        test.
        """
        self.reactor = task.Clock()

        class ProtocolInterpreter:
            dtpInstance = None
        self.protocolInterpreter = ProtocolInterpreter()
        self.factory = ftp.DTPFactory(self.protocolInterpreter, None, self.reactor)

    def test_setTimeout(self):
        """
        L{ftp.DTPFactory.setTimeout} uses the reactor passed to its initializer
        to set up a timed event to time out the DTP setup after the specified
        number of seconds.
        """
        finished = []
        d = self.assertFailure(self.factory.deferred, ftp.PortConnectionError)
        d.addCallback(finished.append)
        self.factory.setTimeout(6)
        self.reactor.advance(5)
        self.assertFalse(finished)
        self.reactor.advance(1)
        self.assertTrue(finished)
        self.assertFalse(self.reactor.calls)

    def test_buildProtocolOnce(self):
        """
        A L{ftp.DTPFactory} instance's C{buildProtocol} method can be used once
        to create a L{ftp.DTP} instance.
        """
        protocol = self.factory.buildProtocol(None)
        self.assertIsInstance(protocol, ftp.DTP)
        self.assertIsNone(self.factory.buildProtocol(None))

    def test_timeoutAfterConnection(self):
        """
        If a timeout has been set up using L{ftp.DTPFactory.setTimeout}, it is
        cancelled by L{ftp.DTPFactory.buildProtocol}.
        """
        self.factory.setTimeout(10)
        self.factory.buildProtocol(None)
        self.assertFalse(self.reactor.calls)

    def test_connectionAfterTimeout(self):
        """
        If L{ftp.DTPFactory.buildProtocol} is called after the timeout
        specified by L{ftp.DTPFactory.setTimeout} has elapsed, L{None} is
        returned.
        """
        d = self.assertFailure(self.factory.deferred, ftp.PortConnectionError)
        self.factory.setTimeout(10)
        self.reactor.advance(10)
        self.assertIsNone(self.factory.buildProtocol(None))
        return d

    def test_timeoutAfterConnectionFailed(self):
        """
        L{ftp.DTPFactory.deferred} fails with L{PortConnectionError} when
        L{ftp.DTPFactory.clientConnectionFailed} is called.  If the timeout
        specified with L{ftp.DTPFactory.setTimeout} expires after that, nothing
        additional happens.
        """
        finished = []
        d = self.assertFailure(self.factory.deferred, ftp.PortConnectionError)
        d.addCallback(finished.append)
        self.factory.setTimeout(10)
        self.assertFalse(finished)
        self.factory.clientConnectionFailed(None, None)
        self.assertTrue(finished)
        self.reactor.advance(10)
        return d

    def test_connectionFailedAfterTimeout(self):
        """
        If L{ftp.DTPFactory.clientConnectionFailed} is called after the timeout
        specified by L{ftp.DTPFactory.setTimeout} has elapsed, nothing beyond
        the normal timeout before happens.
        """
        d = self.assertFailure(self.factory.deferred, ftp.PortConnectionError)
        self.factory.setTimeout(10)
        self.reactor.advance(10)
        self.factory.clientConnectionFailed(None, defer.TimeoutError('foo'))
        return d