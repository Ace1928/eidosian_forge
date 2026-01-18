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
def _joinAddr2(self, interface, addr, join):
    addr = socket.inet_aton(addr)
    interface = socket.inet_aton(interface)
    if join:
        cmd = socket.IP_ADD_MEMBERSHIP
    else:
        cmd = socket.IP_DROP_MEMBERSHIP
    try:
        self.socket.setsockopt(socket.IPPROTO_IP, cmd, addr + interface)
    except OSError as e:
        return failure.Failure(error.MulticastJoinError(addr, interface, *e.args))