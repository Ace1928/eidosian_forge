import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINTR, EINVAL, ENOBUFS, ENOSYS, EPERM, EWOULDBLOCK
from functools import wraps
from zope.interface import implementer
from twisted.internet.protocol import DatagramProtocol
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.pair.tuntap import _IFNAMSIZ, _TUNSETIFF, TunnelFlags, _IInputOutputSystem
from twisted.python.compat import nativeString
@implementer(_IInputOutputSystem)
class MemoryIOSystem:
    """
    An in-memory implementation of basic I/O primitives, useful in the context
    of unit testing as a drop-in replacement for parts of the C{os} module.

    @ivar _devices:
    @ivar _openFiles:
    @ivar permissions:

    @ivar _counter:
    """
    _counter = 8192
    O_RDWR = 1 << 0
    O_NONBLOCK = 1 << 1
    O_CLOEXEC = 1 << 2

    def __init__(self):
        self._devices = {}
        self._openFiles = {}
        self.permissions = {'open', 'ioctl'}

    def getTunnel(self, port):
        """
        Get the L{Tunnel} object associated with the given L{TuntapPort}.

        @param port: A L{TuntapPort} previously initialized using this
            L{MemoryIOSystem}.

        @return: The tunnel object created by a prior use of C{open} on this
            object on the tunnel special device file.
        @rtype: L{Tunnel}
        """
        return self._openFiles[port.fileno()]

    def registerSpecialDevice(self, name, cls):
        """
        Specify a class which will be used to handle I/O to a device of a
        particular name.

        @param name: The filesystem path name of the device.
        @type name: L{bytes}

        @param cls: A class (like L{Tunnel}) to instantiated whenever this
            device is opened.
        """
        self._devices[name] = cls

    @_privileged
    def open(self, name, flags, mode=None):
        """
        A replacement for C{os.open}.  This initializes state in this
        L{MemoryIOSystem} which will be reflected in the behavior of the other
        file descriptor-related methods (eg L{MemoryIOSystem.read},
        L{MemoryIOSystem.write}, etc).

        @param name: A string giving the name of the file to open.
        @type name: C{bytes}

        @param flags: The flags with which to open the file.
        @type flags: C{int}

        @param mode: The mode with which to open the file.
        @type mode: C{int}

        @raise OSError: With C{ENOSYS} if the file is not a recognized special
            device file.

        @return: A file descriptor associated with the newly opened file
            description.
        @rtype: L{int}
        """
        if name in self._devices:
            fd = self._counter
            self._counter += 1
            self._openFiles[fd] = self._devices[name](self, flags, mode)
            return fd
        raise OSError(ENOSYS, 'Function not implemented')

    def read(self, fd, limit):
        """
        Try to read some bytes out of one of the in-memory buffers which may
        previously have been populated by C{write}.

        @see: L{os.read}
        """
        try:
            return self._openFiles[fd].read(limit)
        except KeyError:
            raise OSError(EBADF, 'Bad file descriptor')

    def write(self, fd, data):
        """
        Try to add some bytes to one of the in-memory buffers to be accessed by
        a later C{read} call.

        @see: L{os.write}
        """
        try:
            return self._openFiles[fd].write(data)
        except KeyError:
            raise OSError(EBADF, 'Bad file descriptor')

    def close(self, fd):
        """
        Discard the in-memory buffer and other in-memory state for the given
        file descriptor.

        @see: L{os.close}
        """
        try:
            del self._openFiles[fd]
        except KeyError:
            raise OSError(EBADF, 'Bad file descriptor')

    @_privileged
    def ioctl(self, fd, request, args):
        """
        Perform some configuration change to the in-memory state for the given
        file descriptor.

        @see: L{fcntl.ioctl}
        """
        try:
            tunnel = self._openFiles[fd]
        except KeyError:
            raise OSError(EBADF, 'Bad file descriptor')
        if request != _TUNSETIFF:
            raise OSError(EINVAL, 'Request or args is not valid.')
        name, mode = struct.unpack('%dsH' % (_IFNAMSIZ,), args)
        tunnel.tunnelMode = mode
        tunnel.requestedName = name
        tunnel.name = name[:_IFNAMSIZ - 3] + b'123'
        return struct.pack('%dsH' % (_IFNAMSIZ,), tunnel.name, mode)

    def sendUDP(self, datagram, address):
        """
        Write an ethernet frame containing an ip datagram containing a udp
        datagram containing the given payload, addressed to the given address,
        to a tunnel device previously opened on this I/O system.

        @param datagram: A UDP datagram payload to send.
        @type datagram: L{bytes}

        @param address: The destination to which to send the datagram.
        @type address: L{tuple} of (L{bytes}, L{int})

        @return: A two-tuple giving the address from which gives the address
            from which the datagram was sent.
        @rtype: L{tuple} of (L{bytes}, L{int})
        """
        srcIP = '10.1.2.3'
        srcPort = 21345
        serialized = _ip(src=srcIP, dst=address[0], payload=_udp(src=srcPort, dst=address[1], payload=datagram))
        openFiles = list(self._openFiles.values())
        openFiles[0].addToReadBuffer(serialized)
        return (srcIP, srcPort)

    def receiveUDP(self, fileno, host, port):
        """
        Get a socket-like object which can be used to receive a datagram sent
        from the given address.

        @param fileno: A file descriptor representing a tunnel device which the
            datagram will be received via.
        @type fileno: L{int}

        @param host: The IPv4 address to which the datagram was sent.
        @type host: L{bytes}

        @param port: The UDP port number to which the datagram was sent.
            received.
        @type port: L{int}

        @return: A L{socket.socket}-like object which can be used to receive
            the specified datagram.
        """
        return _FakePort(self, fileno)