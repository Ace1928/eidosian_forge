import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class TimeoutMixinTests(unittest.TestCase):
    """
    Tests for L{policies.TimeoutMixin}.
    """

    def setUp(self):
        """
        Create a testable, deterministic clock and a C{TimeoutTester} instance.
        """
        self.clock = task.Clock()
        self.proto = TimeoutTester(self.clock)

    def test_overriddenCallLater(self):
        """
        Test that the callLater of the clock is used instead of
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        """
        self.proto.setTimeout(10)
        self.assertEqual(len(self.clock.calls), 1)

    def test_timeout(self):
        """
        Check that the protocol does timeout at the time specified by its
        C{timeOut} attribute.
        """
        self.proto.makeConnection(StringTransport())
        self.clock.pump([0, 0.5, 1.0, 1.0])
        self.assertFalse(self.proto.timedOut)
        self.clock.pump([0, 1.0])
        self.assertTrue(self.proto.timedOut)

    def test_noTimeout(self):
        """
        Check that receiving data is delaying the timeout of the connection.
        """
        self.proto.makeConnection(StringTransport())
        self.clock.pump([0, 0.5, 1.0, 1.0])
        self.assertFalse(self.proto.timedOut)
        self.proto.dataReceived(b'hello there')
        self.clock.pump([0, 1.0, 1.0, 0.5])
        self.assertFalse(self.proto.timedOut)
        self.clock.pump([0, 1.0])
        self.assertTrue(self.proto.timedOut)

    def test_resetTimeout(self):
        """
        Check that setting a new value for timeout cancel the previous value
        and install a new timeout.
        """
        self.proto.timeOut = None
        self.proto.makeConnection(StringTransport())
        self.proto.setTimeout(1)
        self.assertEqual(self.proto.timeOut, 1)
        self.clock.pump([0, 0.9])
        self.assertFalse(self.proto.timedOut)
        self.clock.pump([0, 0.2])
        self.assertTrue(self.proto.timedOut)

    def test_cancelTimeout(self):
        """
        Setting the timeout to L{None} cancel any timeout operations.
        """
        self.proto.timeOut = 5
        self.proto.makeConnection(StringTransport())
        self.proto.setTimeout(None)
        self.assertIsNone(self.proto.timeOut)
        self.clock.pump([0, 5, 5, 5])
        self.assertFalse(self.proto.timedOut)

    def test_setTimeoutReturn(self):
        """
        setTimeout should return the value of the previous timeout.
        """
        self.proto.timeOut = 5
        self.assertEqual(self.proto.setTimeout(10), 5)
        self.assertEqual(self.proto.setTimeout(None), 10)
        self.assertIsNone(self.proto.setTimeout(1))
        self.assertEqual(self.proto.timeOut, 1)
        self.proto.setTimeout(None)

    def test_setTimeoutCancleAlreadyCancelled(self):
        """
        When the timeout was already cancelled from an external place,
        calling setTimeout with C{None} to explicitly cancel it will clean
        up the timeout without raising any exception.
        """
        self.proto.setTimeout(3)
        self.clock.getDelayedCalls()[0].cancel()
        self.assertIsNotNone(self.proto.timeOut)
        self.proto.setTimeout(None)
        self.assertIsNone(self.proto.timeOut)