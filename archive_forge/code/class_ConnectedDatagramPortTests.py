import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class ConnectedDatagramPortTests(WarningCheckerTestCase):
    """
    Test connected datagram UNIX sockets.
    """
    if skipSockets is not None:
        skip = skipSockets

    def test_connectionFailedDoesntCallLoseConnection(self):
        """
        L{ConnectedDatagramPort} does not call the deprecated C{loseConnection}
        in L{ConnectedDatagramPort.connectionFailed}.
        """

        def loseConnection():
            """
            Dummy C{loseConnection} method. C{loseConnection} is deprecated and
            should not get called.
            """
            self.fail('loseConnection is deprecated and should not get called.')
        port = unix.ConnectedDatagramPort(None, ClientProto())
        port.loseConnection = loseConnection
        port.connectionFailed('goodbye')

    def test_connectionFailedCallsStopListening(self):
        """
        L{ConnectedDatagramPort} calls L{ConnectedDatagramPort.stopListening}
        instead of the deprecated C{loseConnection} in
        L{ConnectedDatagramPort.connectionFailed}.
        """
        self.called = False

        def stopListening():
            """
            Dummy C{stopListening} method.
            """
            self.called = True
        port = unix.ConnectedDatagramPort(None, ClientProto())
        port.stopListening = stopListening
        port.connectionFailed('goodbye')
        self.assertTrue(self.called)