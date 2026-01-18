from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorSocket(Interface):
    """
    Methods which allow a reactor to use externally created sockets.

    For example, to use C{adoptStreamPort} to implement behavior equivalent
    to that of L{IReactorTCP.listenTCP}, you might write code like this::

        from socket import SOMAXCONN, AF_INET, SOCK_STREAM, socket
        portSocket = socket(AF_INET, SOCK_STREAM)
        # Set FD_CLOEXEC on port, left as an exercise.  Then make it into a
        # non-blocking listening port:
        portSocket.setblocking(False)
        portSocket.bind(('192.168.1.2', 12345))
        portSocket.listen(SOMAXCONN)

        # Now have the reactor use it as a TCP port
        port = reactor.adoptStreamPort(
            portSocket.fileno(), AF_INET, YourFactory())

        # portSocket itself is no longer necessary, and needs to be cleaned
        # up by us.
        portSocket.close()

        # Whenever the server is no longer needed, stop it as usual.
        stoppedDeferred = port.stopListening()

    Another potential use is to inherit a listening descriptor from a parent
    process (for example, systemd or launchd), or to receive one over a UNIX
    domain socket.

    Some plans for extending this interface exist.  See:

        - U{http://twistedmatrix.com/trac/ticket/6594}: AF_UNIX SOCK_DGRAM ports
    """

    def adoptStreamPort(fileDescriptor: int, addressFamily: 'AddressFamily', factory: 'ServerFactory') -> 'IListeningPort':
        """
        Add an existing listening I{SOCK_STREAM} socket to the reactor to
        monitor for new connections to accept and handle.

        @param fileDescriptor: A file descriptor associated with a socket which
            is already bound to an address and marked as listening.  The socket
            must be set non-blocking.  Any additional flags (for example,
            close-on-exec) must also be set by application code.  Application
            code is responsible for closing the file descriptor, which may be
            done as soon as C{adoptStreamPort} returns.
        @param addressFamily: The address family (or I{domain}) of the socket.
            For example, L{socket.AF_INET6}.
        @param factory: A L{ServerFactory} instance to use to create new
            protocols to handle connections accepted via this socket.

        @return: An object providing L{IListeningPort}.

        @raise twisted.internet.error.UnsupportedAddressFamily: If the
            given address family is not supported by this reactor, or
            not supported with the given socket type.
        @raise twisted.internet.error.UnsupportedSocketType: If the
            given socket type is not supported by this reactor, or not
            supported with the given socket type.
        """

    def adoptStreamConnection(fileDescriptor: int, addressFamily: 'AddressFamily', factory: 'ServerFactory') -> None:
        """
        Add an existing connected I{SOCK_STREAM} socket to the reactor to
        monitor for data.

        Note that the given factory won't have its C{startFactory} and
        C{stopFactory} methods called, as there is no sensible time to call
        them in this situation.

        @param fileDescriptor: A file descriptor associated with a socket which
            is already connected.  The socket must be set non-blocking.  Any
            additional flags (for example, close-on-exec) must also be set by
            application code.  Application code is responsible for closing the
            file descriptor, which may be done as soon as
            C{adoptStreamConnection} returns.
        @param addressFamily: The address family (or I{domain}) of the socket.
            For example, L{socket.AF_INET6}.
        @param factory: A L{ServerFactory} instance to use to create a new
            protocol to handle the connection via this socket.

        @raise UnsupportedAddressFamily: If the given address family is not
            supported by this reactor, or not supported with the given socket
            type.
        @raise UnsupportedSocketType: If the given socket type is not supported
            by this reactor, or not supported with the given socket type.
        """

    def adoptDatagramPort(fileDescriptor: int, addressFamily: 'AddressFamily', protocol: 'DatagramProtocol', maxPacketSize: int) -> 'IListeningPort':
        """
        Add an existing listening I{SOCK_DGRAM} socket to the reactor to
        monitor for read and write readiness.

        @param fileDescriptor: A file descriptor associated with a socket which
            is already bound to an address and marked as listening.  The socket
            must be set non-blocking.  Any additional flags (for example,
            close-on-exec) must also be set by application code.  Application
            code is responsible for closing the file descriptor, which may be
            done as soon as C{adoptDatagramPort} returns.
        @param addressFamily: The address family or I{domain} of the socket.
            For example, L{socket.AF_INET6}.
        @param protocol: A L{DatagramProtocol} instance to connect to
            a UDP transport.
        @param maxPacketSize: The maximum packet size to accept.

        @return: An object providing L{IListeningPort}.

        @raise UnsupportedAddressFamily: If the given address family is not
            supported by this reactor, or not supported with the given socket
            type.
        @raise UnsupportedSocketType: If the given socket type is not supported
            by this reactor, or not supported with the given socket type.
        """