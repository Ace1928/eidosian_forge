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
class Tunnel:
    """
    An in-memory implementation of a tun or tap device.

    @cvar _DEVICE_NAME: A string representing the conventional filesystem entry
        for the tunnel factory character special device.
    @type _DEVICE_NAME: C{bytes}
    """
    _DEVICE_NAME = b'/dev/net/tun'
    EAGAIN_STYLE = IOError(EAGAIN, 'Resource temporarily unavailable')
    EWOULDBLOCK_STYLE = OSError(EWOULDBLOCK, 'Operation would block')
    EINTR_STYLE = IOError(EINTR, 'Interrupted function call')
    nonBlockingExceptionStyle = EAGAIN_STYLE
    SEND_BUFFER_SIZE = 1024

    def __init__(self, system, openFlags, fileMode):
        """
        @param system: An L{_IInputOutputSystem} provider to use to perform I/O.

        @param openFlags: Any flags to apply when opening the tunnel device.
            See C{os.O_*}.

        @type openFlags: L{int}

        @param fileMode: ignored
        """
        self.system = system
        self.openFlags = openFlags
        self.tunnelMode = None
        self.requestedName = None
        self.name = None
        self.readBuffer = deque()
        self.writeBuffer = deque()
        self.pendingSignals = deque()

    @property
    def blocking(self):
        """
        If the file descriptor for this tunnel is open in blocking mode,
        C{True}.  C{False} otherwise.
        """
        return not self.openFlags & self.system.O_NONBLOCK

    @property
    def closeOnExec(self):
        """
        If the file descriptor for this tunnel is marked as close-on-exec,
        C{True}.  C{False} otherwise.
        """
        return bool(self.openFlags & self.system.O_CLOEXEC)

    def addToReadBuffer(self, datagram):
        """
        Deliver a datagram to this tunnel's read buffer.  This makes it
        available to be read later using the C{read} method.

        @param datagram: The IPv4 datagram to deliver.  If the mode of this
            tunnel is TAP then ethernet framing will be added automatically.
        @type datagram: L{bytes}
        """
        if self.tunnelMode & TunnelFlags.IFF_TAP.value:
            datagram = _ethernet(src=b'\x00' * 6, dst=b'\xff' * 6, protocol=_IPv4, payload=datagram)
        self.readBuffer.append(datagram)

    def read(self, limit):
        """
        Read a datagram out of this tunnel.

        @param limit: The maximum number of bytes from the datagram to return.
            If the next datagram is larger than this, extra bytes are dropped
            and lost forever.
        @type limit: L{int}

        @raise OSError: Any of the usual I/O problems can result in this
            exception being raised with some particular error number set.

        @raise IOError: Any of the usual I/O problems can result in this
            exception being raised with some particular error number set.

        @return: The datagram which was read from the tunnel.  If the tunnel
            mode does not include L{TunnelFlags.IFF_NO_PI} then the datagram is
            prefixed with a 4 byte PI header.
        @rtype: L{bytes}
        """
        if self.readBuffer:
            if self.tunnelMode & TunnelFlags.IFF_NO_PI.value:
                header = b''
            else:
                header = b'\x00' * _PI_SIZE
                limit -= 4
            return header + self.readBuffer.popleft()[:limit]
        elif self.blocking:
            raise NotImplementedError()
        else:
            raise self.nonBlockingExceptionStyle

    def write(self, datagram):
        """
        Write a datagram into this tunnel.

        @param datagram: The datagram to write.
        @type datagram: L{bytes}

        @raise IOError: Any of the usual I/O problems can result in this
            exception being raised with some particular error number set.

        @return: The number of bytes of the datagram which were written.
        @rtype: L{int}
        """
        if self.pendingSignals:
            self.pendingSignals.popleft()
            raise OSError(EINTR, 'Interrupted system call')
        if len(datagram) > self.SEND_BUFFER_SIZE:
            raise OSError(ENOBUFS, 'No buffer space available')
        self.writeBuffer.append(datagram)
        return len(datagram)