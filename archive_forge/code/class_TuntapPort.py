import errno
import fcntl
import os
import platform
import struct
import warnings
from collections import namedtuple
from typing import Tuple
from zope.interface import Attribute, Interface, implementer
from constantly import FlagConstant, Flags
from incremental import Version
from twisted.internet import abstract, defer, error, interfaces, task
from twisted.pair import ethernet, raw
from twisted.python import log
from twisted.python.deprecate import deprecated
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import FancyEqMixin, FancyStrMixin
@implementer(interfaces.IListeningPort)
class TuntapPort(abstract.FileDescriptor):
    """
    A Port that reads and writes packets from/to a TUN/TAP-device.
    """
    maxThroughput = 256 * 1024

    def __init__(self, interface, proto, maxPacketSize=8192, reactor=None, system=None):
        if ethernet.IEthernetProtocol.providedBy(proto):
            self.ethernet = 1
            self._mode = TunnelFlags.IFF_TAP
        else:
            self.ethernet = 0
            self._mode = TunnelFlags.IFF_TUN
            assert raw.IRawPacketProtocol.providedBy(proto)
        if system is None:
            system = _RealSystem()
        self._system = system
        abstract.FileDescriptor.__init__(self, reactor)
        self.interface = interface
        self.protocol = proto
        self.maxPacketSize = maxPacketSize
        logPrefix = self._getLogPrefix(self.protocol)
        self.logstr = f'{logPrefix} ({self._mode.name})'

    def __repr__(self) -> str:
        args: Tuple[str, ...] = (fullyQualifiedName(self.protocol.__class__),)
        if self.connected:
            args = args + ('',)
        else:
            args = args + ('not ',)
        args = args + (self._mode.name, self.interface)
        return '<%s %slistening on %s/%s>' % args

    def startListening(self):
        """
        Create and bind my socket, and begin listening on it.

        This must be called after creating a server to begin listening on the
        specified tunnel.
        """
        self._bindSocket()
        self.protocol.makeConnection(self)
        self.startReading()

    def _openTunnel(self, name, mode):
        """
        Open the named tunnel using the given mode.

        @param name: The name of the tunnel to open.
        @type name: L{bytes}

        @param mode: Flags from L{TunnelFlags} with exactly one of
            L{TunnelFlags.IFF_TUN} or L{TunnelFlags.IFF_TAP} set.

        @return: A L{_TunnelDescription} representing the newly opened tunnel.
        """
        flags = self._system.O_RDWR | self._system.O_CLOEXEC | self._system.O_NONBLOCK
        config = struct.pack('%dsH' % (_IFNAMSIZ,), name, mode.value)
        fileno = self._system.open(_TUN_KO_PATH, flags)
        result = self._system.ioctl(fileno, _TUNSETIFF, config)
        return _TunnelDescription(fileno, result[:_IFNAMSIZ].strip(b'\x00'))

    def _bindSocket(self):
        """
        Open the tunnel.
        """
        log.msg(format='%(protocol)s starting on %(interface)s', protocol=self.protocol.__class__, interface=self.interface)
        try:
            fileno, interface = self._openTunnel(self.interface, self._mode | TunnelFlags.IFF_NO_PI)
        except OSError as e:
            raise error.CannotListenError(None, self.interface, e)
        self.interface = interface
        self._fileno = fileno
        self.connected = 1

    def fileno(self):
        return self._fileno

    def doRead(self):
        """
        Called when my socket is ready for reading.
        """
        read = 0
        while read < self.maxThroughput:
            try:
                data = self._system.read(self._fileno, self.maxPacketSize)
            except OSError as e:
                if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EINTR):
                    return
                else:
                    raise
            except BaseException:
                raise
            read += len(data)
            try:
                self.protocol.datagramReceived(data, partial=0)
            except BaseException:
                cls = fullyQualifiedName(self.protocol.__class__)
                log.err(None, f'Unhandled exception from {cls}.datagramReceived')

    def write(self, datagram):
        """
        Write the given data as a single datagram.

        @param datagram: The data that will make up the complete datagram to be
            written.
        @type datagram: L{bytes}
        """
        try:
            return self._system.write(self._fileno, datagram)
        except OSError as e:
            if e.errno == errno.EINTR:
                return self.write(datagram)
            raise

    def writeSequence(self, seq):
        """
        Write a datagram constructed from a L{list} of L{bytes}.

        @param seq: The data that will make up the complete datagram to be
            written.
        @type seq: L{list} of L{bytes}
        """
        self.write(b''.join(seq))

    def stopListening(self):
        """
        Stop accepting connections on this port.

        This will shut down my socket and call self.connectionLost().

        @return: A L{Deferred} that fires when this port has stopped.
        """
        self.stopReading()
        if self.disconnecting:
            return self._stoppedDeferred
        elif self.connected:
            self._stoppedDeferred = task.deferLater(self.reactor, 0, self.connectionLost)
            self.disconnecting = True
            return self._stoppedDeferred
        else:
            return defer.succeed(None)

    @deprecated(Version('Twisted', 14, 0, 0), stopListening)
    def loseConnection(self):
        """
        Close this tunnel.  Use L{TuntapPort.stopListening} instead.
        """
        self.stopListening().addErrback(log.err)

    def connectionLost(self, reason=None):
        """
        Cleans up my socket.

        @param reason: Ignored.  Do not use this.
        """
        log.msg('(Tuntap %s Closed)' % self.interface)
        abstract.FileDescriptor.connectionLost(self, reason)
        self.protocol.doStop()
        self.connected = 0
        self._system.close(self._fileno)
        self._fileno = -1

    def logPrefix(self):
        """
        Returns the name of my class, to prefix log entries with.
        """
        return self.logstr

    def getHost(self):
        """
        Get the local address of this L{TuntapPort}.

        @return: A L{TunnelAddress} which describes the tunnel device to which
            this object is bound.
        @rtype: L{TunnelAddress}
        """
        return TunnelAddress(self._mode, self.interface)