import os
import socket
import stat
import struct
from errno import EAGAIN, ECONNREFUSED, EINTR, EMSGSIZE, ENOBUFS, EWOULDBLOCK
from typing import Optional, Type
from zope.interface import implementedBy, implementer, implementer_only
from twisted.internet import address, base, error, interfaces, main, protocol, tcp, udp
from twisted.internet.abstract import FileDescriptor
from twisted.python import failure, lockfile, log, reflect
from twisted.python.compat import lazyByteSlice
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.util import untilConcludes
@implementer_only(interfaces.IUNIXDatagramConnectedTransport, *implementedBy(base.BasePort))
class ConnectedDatagramPort(DatagramPort):
    """
    A connected datagram UNIX socket.
    """

    def __init__(self, addr, proto, maxPacketSize=8192, mode=438, bindAddress=None, reactor=None):
        assert isinstance(proto, protocol.ConnectedDatagramProtocol)
        DatagramPort.__init__(self, bindAddress, proto, maxPacketSize, mode, reactor)
        self.remoteaddr = addr

    def startListening(self):
        try:
            self._bindSocket()
            self.socket.connect(self.remoteaddr)
            self._connectToProtocol()
        except BaseException:
            self.connectionFailed(failure.Failure())

    def connectionFailed(self, reason):
        """
        Called when a connection fails. Stop listening on the socket.

        @type reason: L{Failure}
        @param reason: Why the connection failed.
        """
        self.stopListening()
        self.protocol.connectionFailed(reason)
        del self.protocol

    def doRead(self):
        """
        Called when my socket is ready for reading.
        """
        read = 0
        while read < self.maxThroughput:
            try:
                data, addr = self.socket.recvfrom(self.maxPacketSize)
                read += len(data)
                self.protocol.datagramReceived(data)
            except OSError as se:
                no = se.args[0]
                if no in (EAGAIN, EINTR, EWOULDBLOCK):
                    return
                if no == ECONNREFUSED:
                    self.protocol.connectionRefused()
                else:
                    raise
            except BaseException:
                log.deferr()

    def write(self, data):
        """
        Write a datagram.
        """
        try:
            return self.socket.send(data)
        except OSError as se:
            no = se.args[0]
            if no == EINTR:
                return self.write(data)
            elif no == EMSGSIZE:
                raise error.MessageLengthError('message too long')
            elif no == ECONNREFUSED:
                self.protocol.connectionRefused()
            elif no == EAGAIN:
                pass
            else:
                raise

    def getPeer(self):
        return address.UNIXAddress(self.remoteaddr)