import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
@skipIf(not interfaces.IReactorUNIXDatagram(reactor, None), 'This reactor does not support UNIX datagram sockets')
class DatagramUnixSocketTests(unittest.TestCase):
    """
    Test datagram UNIX sockets.
    """

    def test_exchange(self):
        """
        Test that a datagram can be sent to and received by a server and vice
        versa.
        """
        clientaddr = self.mktemp()
        serveraddr = self.mktemp()
        sp = ServerProto()
        cp = ClientProto()
        s = reactor.listenUNIXDatagram(serveraddr, sp)
        self.addCleanup(s.stopListening)
        c = reactor.connectUNIXDatagram(serveraddr, cp, bindAddress=clientaddr)
        self.addCleanup(c.stopListening)
        d = defer.gatherResults([sp.deferredStarted, cp.deferredStarted])

        def write(ignored):
            cp.transport.write(b'hi')
            return defer.gatherResults([sp.deferredGotWhat, cp.deferredGotBack])

        def _cbTestExchange(ignored):
            self.assertEqual(b'hi', sp.gotwhat)
            self.assertEqual(clientaddr, sp.gotfrom)
            self.assertEqual(b'hi back', cp.gotback)
        d.addCallback(write)
        d.addCallback(_cbTestExchange)
        return d

    def test_cannotListen(self):
        """
        L{IReactorUNIXDatagram.listenUNIXDatagram} raises
        L{error.CannotListenError} if the unix socket specified is already in
        use.
        """
        addr = self.mktemp()
        p = ServerProto()
        s = reactor.listenUNIXDatagram(addr, p)
        self.assertRaises(error.CannotListenError, reactor.listenUNIXDatagram, addr, p)
        s.stopListening()
        os.unlink(addr)

    def _reprTest(self, serverProto, protocolName):
        """
        Test the C{__str__} and C{__repr__} implementations of a UNIX datagram
        port when used with the given protocol.
        """
        filename = self.mktemp()
        unixPort = reactor.listenUNIXDatagram(filename, serverProto)
        connectedString = f'<{protocolName} on {filename!r}>'
        self.assertEqual(repr(unixPort), connectedString)
        self.assertEqual(str(unixPort), connectedString)
        stopDeferred = defer.maybeDeferred(unixPort.stopListening)

        def stoppedListening(ign):
            unconnectedString = f'<{protocolName} (not listening)>'
            self.assertEqual(repr(unixPort), unconnectedString)
            self.assertEqual(str(unixPort), unconnectedString)
        stopDeferred.addCallback(stoppedListening)
        return stopDeferred

    def test_reprWithNewStyleProtocol(self):
        """
        The two string representations of the L{IListeningPort} returned by
        L{IReactorUNIXDatagram.listenUNIXDatagram} contains the name of the
        new-style protocol class being used and the filename on which the port
        is listening or indicates that the port is not listening.
        """

        class NewStyleProtocol:

            def makeConnection(self, transport):
                pass

            def doStop(self):
                pass
        self.assertIsInstance(NewStyleProtocol, type)
        return self._reprTest(NewStyleProtocol(), 'twisted.test.test_unix.NewStyleProtocol')