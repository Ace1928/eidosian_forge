import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
class AdoptDatagramPortErrorsTestsBuilder(ReactorBuilder):
    """
    Builder for testing L{IReactorSocket.adoptDatagramPort} implementations.
    """
    requiredInterfaces = [IReactorSocket]

    def test_invalidDescriptor(self):
        """
        An implementation of L{IReactorSocket.adoptDatagramPort} raises
        L{socket.error} if passed an integer which is not associated with a
        socket.
        """
        reactor = self.buildReactor()
        probe = socket.socket()
        fileno = probe.fileno()
        probe.close()
        exc = self.assertRaises(socket.error, reactor.adoptDatagramPort, fileno, socket.AF_INET, DatagramProtocol())
        if platform.isWindows():
            self.assertEqual(exc.args[0], errno.WSAENOTSOCK)
        else:
            self.assertEqual(exc.args[0], errno.EBADF)

    def test_invalidAddressFamily(self):
        """
        An implementation of L{IReactorSocket.adoptDatagramPort} raises
        L{UnsupportedAddressFamily} if passed an address family it does not
        support.
        """
        reactor = self.buildReactor()
        port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(port.close)
        arbitrary = 2 ** 16 + 7
        self.assertRaises(UnsupportedAddressFamily, reactor.adoptDatagramPort, port.fileno(), arbitrary, DatagramProtocol())

    def test_stopOnlyCloses(self):
        """
        When the L{IListeningPort} returned by
        L{IReactorSocket.adoptDatagramPort} is stopped using
        C{stopListening}, the underlying socket is closed but not
        shutdown.  This allows another process which still has a
        reference to it to continue reading and writing to it.
        """
        reactor = self.buildReactor()
        portSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(portSocket.close)
        portSocket.bind(('127.0.0.1', 0))
        portSocket.setblocking(False)
        port = reactor.adoptDatagramPort(portSocket.fileno(), portSocket.family, DatagramProtocol())
        d = port.stopListening()

        def stopped(ignored):
            exc = self.assertRaises(socket.error, portSocket.recvfrom, 1)
            if platform.isWindows():
                self.assertEqual(exc.args[0], errno.WSAEWOULDBLOCK)
            else:
                self.assertEqual(exc.args[0], errno.EAGAIN)
        d.addCallback(stopped)
        d.addErrback(err, 'Failed to read on original port.')
        needsRunningReactor(reactor, lambda: d.addCallback(lambda ignored: reactor.stop()))
        reactor.run()