from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def _cancelConnectTimeoutTest(self, connect):
    """
        Like L{_cancelConnectTest}, but for the case where the L{Deferred} is
        cancelled after the connection is set up but before it is fired with the
        resulting protocol instance.
        """
    reactor = MemoryReactorClock()
    cc = ClientCreator(reactor, Protocol)
    d = connect(reactor, cc)
    connector = reactor.connectors.pop()
    self.assertEqual(len(reactor.getDelayedCalls()), 1)
    d.cancel()
    self.assertEqual(reactor.getDelayedCalls(), [])
    self.assertTrue(connector._disconnected)
    return self.assertFailure(d, CancelledError)