import socket
import struct
import warnings
from typing import Optional
from zope.interface import implementer
from twisted.python.runtime import platformType
from twisted.internet import abstract, address, base, defer, error, interfaces
from twisted.python import failure, log
@classmethod
def _fromListeningDescriptor(cls, reactor, fd, addressFamily, protocol, maxPacketSize):
    """
        Create a new L{Port} based on an existing listening
        I{SOCK_DGRAM} socket.

        @param reactor: A reactor which will notify this L{Port} when
            its socket is ready for reading or writing. Defaults to
            L{None}, ie the default global reactor.
        @type reactor: L{interfaces.IReactorFDSet}

        @param fd: An integer file descriptor associated with a listening
            socket.  The socket must be in non-blocking mode.  Any additional
            attributes desired, such as I{FD_CLOEXEC}, must also be set already.
        @type fd: L{int}

        @param addressFamily: The address family (sometimes called I{domain}) of
            the existing socket.  For example, L{socket.AF_INET}.
        @type addressFamily: L{int}

        @param protocol: A C{DatagramProtocol} instance which will be
            connected to the C{port}.
        @type protocol: L{twisted.internet.protocol.DatagramProtocol}

        @param maxPacketSize: The maximum packet size to accept.
        @type maxPacketSize: L{int}

        @return: A new instance of C{cls} wrapping the socket given by C{fd}.
        @rtype: L{Port}
        """
    port = socket.fromfd(fd, addressFamily, cls.socketType)
    interface = port.getsockname()[0]
    self = cls(None, protocol, interface=interface, reactor=reactor, maxPacketSize=maxPacketSize)
    self._preexistingSocket = port
    return self