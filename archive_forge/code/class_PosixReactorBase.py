import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (
@implementer(IReactorTCP, IReactorUDP, IReactorMulticast)
class PosixReactorBase(_DisconnectSelectableMixin, ReactorBase):
    """
    A basis for reactors that use file descriptors.

    @ivar _childWaker: L{None} or a reference to the L{_SIGCHLDWaker}
        which is used to properly notice child process termination.
    """
    _childWaker = None

    def _wakerFactory(self) -> _IWaker:
        return _Waker()

    def installWaker(self):
        """
        Install a `waker' to allow threads and signals to wake up the IO thread.

        We use the self-pipe trick (http://cr.yp.to/docs/selfpipe.html) to wake
        the reactor. On Windows we use a pair of sockets.
        """
        if not self.waker:
            self.waker = self._wakerFactory()
            self._internalReaders.add(self.waker)
            self.addReader(self.waker)

    def _signalsFactory(self) -> SignalHandling:
        """
        Customize reactor signal handling to support child processes on POSIX
        platforms.
        """
        baseHandling = super()._signalsFactory()
        if platformType == 'posix':
            return _MultiSignalHandling((baseHandling, _ChildSignalHandling(self._addInternalReader, self._removeInternalReader)))
        return baseHandling

    def spawnProcess(self, processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        if platformType == 'posix':
            if usePTY:
                if childFDs is not None:
                    raise ValueError('Using childFDs is not supported with usePTY=True.')
                return process.PTYProcess(self, executable, args, env, path, processProtocol, uid, gid, usePTY)
            else:
                return process.Process(self, executable, args, env, path, processProtocol, uid, gid, childFDs)
        elif platformType == 'win32':
            if uid is not None:
                raise ValueError('Setting UID is unsupported on this platform.')
            if gid is not None:
                raise ValueError('Setting GID is unsupported on this platform.')
            if usePTY:
                raise ValueError('The usePTY parameter is not supported on Windows.')
            if childFDs:
                raise ValueError('Customizing childFDs is not supported on Windows.')
            if win32process:
                from twisted.internet._dumbwin32proc import Process
                return Process(self, processProtocol, executable, args, env, path)
            else:
                raise NotImplementedError('spawnProcess not available since pywin32 is not installed.')
        else:
            raise NotImplementedError('spawnProcess only available on Windows or POSIX.')

    def listenUDP(self, port, protocol, interface='', maxPacketSize=8192):
        """Connects a given L{DatagramProtocol} to the given numeric UDP port.

        @returns: object conforming to L{IListeningPort}.
        """
        p = udp.Port(port, protocol, interface, maxPacketSize, self)
        p.startListening()
        return p

    def listenMulticast(self, port, protocol, interface='', maxPacketSize=8192, listenMultiple=False):
        """Connects a given DatagramProtocol to the given numeric UDP port.

        EXPERIMENTAL.

        @returns: object conforming to IListeningPort.
        """
        p = udp.MulticastPort(port, protocol, interface, maxPacketSize, self, listenMultiple)
        p.startListening()
        return p

    def connectUNIX(self, address, factory, timeout=30, checkPID=0):
        assert unixEnabled, 'UNIX support is not present'
        c = unix.Connector(address, factory, timeout, self, checkPID)
        c.connect()
        return c

    def listenUNIX(self, address, factory, backlog=50, mode=438, wantPID=0):
        assert unixEnabled, 'UNIX support is not present'
        p = unix.Port(address, factory, backlog, mode, self, wantPID)
        p.startListening()
        return p

    def listenUNIXDatagram(self, address, protocol, maxPacketSize=8192, mode=438):
        """
        Connects a given L{DatagramProtocol} to the given path.

        EXPERIMENTAL.

        @returns: object conforming to L{IListeningPort}.
        """
        assert unixEnabled, 'UNIX support is not present'
        p = unix.DatagramPort(address, protocol, maxPacketSize, mode, self)
        p.startListening()
        return p

    def connectUNIXDatagram(self, address, protocol, maxPacketSize=8192, mode=438, bindAddress=None):
        """
        Connects a L{ConnectedDatagramProtocol} instance to a path.

        EXPERIMENTAL.
        """
        assert unixEnabled, 'UNIX support is not present'
        p = unix.ConnectedDatagramPort(address, protocol, maxPacketSize, mode, bindAddress, self)
        p.startListening()
        return p
    if unixEnabled:
        _supportedAddressFamilies: Sequence[socket.AddressFamily] = (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX)
    else:
        _supportedAddressFamilies = (socket.AF_INET, socket.AF_INET6)

    def adoptStreamPort(self, fileDescriptor, addressFamily, factory):
        """
        Create a new L{IListeningPort} from an already-initialized socket.

        This just dispatches to a suitable port implementation (eg from
        L{IReactorTCP}, etc) based on the specified C{addressFamily}.

        @see: L{twisted.internet.interfaces.IReactorSocket.adoptStreamPort}
        """
        if addressFamily not in self._supportedAddressFamilies:
            raise error.UnsupportedAddressFamily(addressFamily)
        if unixEnabled and addressFamily == socket.AF_UNIX:
            p = unix.Port._fromListeningDescriptor(self, fileDescriptor, factory)
        else:
            p = tcp.Port._fromListeningDescriptor(self, fileDescriptor, addressFamily, factory)
        p.startListening()
        return p

    def adoptStreamConnection(self, fileDescriptor, addressFamily, factory):
        """
        @see:
            L{twisted.internet.interfaces.IReactorSocket.adoptStreamConnection}
        """
        if addressFamily not in self._supportedAddressFamilies:
            raise error.UnsupportedAddressFamily(addressFamily)
        if unixEnabled and addressFamily == socket.AF_UNIX:
            return unix.Server._fromConnectedSocket(fileDescriptor, factory, self)
        else:
            return tcp.Server._fromConnectedSocket(fileDescriptor, addressFamily, factory, self)

    def adoptDatagramPort(self, fileDescriptor, addressFamily, protocol, maxPacketSize=8192):
        if addressFamily not in (socket.AF_INET, socket.AF_INET6):
            raise error.UnsupportedAddressFamily(addressFamily)
        p = udp.Port._fromListeningDescriptor(self, fileDescriptor, addressFamily, protocol, maxPacketSize=maxPacketSize)
        p.startListening()
        return p

    def listenTCP(self, port, factory, backlog=50, interface=''):
        p = tcp.Port(port, factory, backlog, interface, self)
        p.startListening()
        return p

    def connectTCP(self, host, port, factory, timeout=30, bindAddress=None):
        c = tcp.Connector(host, port, factory, timeout, bindAddress, self)
        c.connect()
        return c

    def connectSSL(self, host, port, factory, contextFactory, timeout=30, bindAddress=None):
        if tls is not None:
            tlsFactory = tls.TLSMemoryBIOFactory(contextFactory, True, factory)
            return self.connectTCP(host, port, tlsFactory, timeout, bindAddress)
        elif ssl is not None:
            c = ssl.Connector(host, port, factory, contextFactory, timeout, bindAddress, self)
            c.connect()
            return c
        else:
            assert False, 'SSL support is not present'

    def listenSSL(self, port, factory, contextFactory, backlog=50, interface=''):
        if tls is not None:
            tlsFactory = tls.TLSMemoryBIOFactory(contextFactory, False, factory)
            port = self.listenTCP(port, tlsFactory, backlog, interface)
            port._type = 'TLS'
            return port
        elif ssl is not None:
            p = ssl.Port(port, factory, contextFactory, backlog, interface, self)
            p.startListening()
            return p
        else:
            assert False, 'SSL support is not present'

    def _removeAll(self, readers, writers):
        """
        Remove all readers and writers, and list of removed L{IReadDescriptor}s
        and L{IWriteDescriptor}s.

        Meant for calling from subclasses, to implement removeAll, like::

          def removeAll(self):
              return self._removeAll(self._reads, self._writes)

        where C{self._reads} and C{self._writes} are iterables.
        """
        removedReaders = set(readers) - self._internalReaders
        for reader in removedReaders:
            self.removeReader(reader)
        removedWriters = set(writers)
        for writer in removedWriters:
            self.removeWriter(writer)
        return list(removedReaders | removedWriters)