import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
class AdoptStreamPortErrorsTestsBuilder(ReactorBuilder):
    """
    Builder for testing L{IReactorSocket.adoptStreamPort} implementations.

    Generally only tests for failure cases are found here.  Success cases for
    this interface are tested elsewhere.  For example, the success case for
    I{AF_INET} is in L{twisted.internet.test.test_tcp}, since that case should
    behave exactly the same as L{IReactorTCP.listenTCP}.
    """
    requiredInterfaces = [IReactorSocket]

    def test_invalidDescriptor(self):
        """
        An implementation of L{IReactorSocket.adoptStreamPort} raises
        L{socket.error} if passed an integer which is not associated with a
        socket.
        """
        reactor = self.buildReactor()
        probe = socket.socket()
        fileno = probe.fileno()
        probe.close()
        exc = self.assertRaises(socket.error, reactor.adoptStreamPort, fileno, socket.AF_INET, ServerFactory())
        if platform.isWindows():
            self.assertEqual(exc.args[0], errno.WSAENOTSOCK)
        else:
            self.assertEqual(exc.args[0], errno.EBADF)

    def test_invalidAddressFamily(self):
        """
        An implementation of L{IReactorSocket.adoptStreamPort} raises
        L{UnsupportedAddressFamily} if passed an address family it does not
        support.
        """
        reactor = self.buildReactor()
        port = socket.socket()
        port.bind(('127.0.0.1', 0))
        port.listen(1)
        self.addCleanup(port.close)
        arbitrary = 2 ** 16 + 7
        self.assertRaises(UnsupportedAddressFamily, reactor.adoptStreamPort, port.fileno(), arbitrary, ServerFactory())

    def test_stopOnlyCloses(self):
        """
        When the L{IListeningPort} returned by
        L{IReactorSocket.adoptStreamPort} is stopped using
        C{stopListening}, the underlying socket is closed but not
        shutdown.  This allows another process which still has a
        reference to it to continue accepting connections over it.
        """
        reactor = self.buildReactor()
        portSocket = socket.socket()
        self.addCleanup(portSocket.close)
        portSocket.bind(('127.0.0.1', 0))
        portSocket.listen(1)
        portSocket.setblocking(False)
        port = reactor.adoptStreamPort(portSocket.fileno(), portSocket.family, ServerFactory())
        d = port.stopListening()

        def stopped(ignored):
            exc = self.assertRaises(socket.error, portSocket.accept)
            if platform.isWindows():
                self.assertEqual(exc.args[0], errno.WSAEWOULDBLOCK)
            else:
                self.assertEqual(exc.args[0], errno.EAGAIN)
        d.addCallback(stopped)
        d.addErrback(err, 'Failed to accept on original port.')
        needsRunningReactor(reactor, lambda: d.addCallback(lambda ignored: reactor.stop()))
        reactor.run()