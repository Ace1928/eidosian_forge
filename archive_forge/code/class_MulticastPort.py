import errno
import socket
import struct
import warnings
from typing import Optional
from zope.interface import implementer
from twisted.internet import address, defer, error, interfaces
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.iocpreactor import abstract, iocpsupport as _iocp
from twisted.internet.iocpreactor.const import (
from twisted.internet.iocpreactor.interfaces import IReadWriteHandle
from twisted.python import failure, log
@implementer(interfaces.IMulticastTransport)
class MulticastPort(MulticastMixin, Port):
    """
    UDP Port that supports multicasting.
    """

    def __init__(self, port, proto, interface='', maxPacketSize=8192, reactor=None, listenMultiple=False):
        Port.__init__(self, port, proto, interface, maxPacketSize, reactor)
        self.listenMultiple = listenMultiple

    def createSocket(self):
        skt = Port.createSocket(self)
        if self.listenMultiple:
            skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, 'SO_REUSEPORT'):
                skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        return skt