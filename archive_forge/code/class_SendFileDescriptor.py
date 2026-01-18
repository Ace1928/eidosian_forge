from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
class SendFileDescriptor(ConnectableProtocol):
    """
    L{SendFileDescriptorAndBytes} sends a file descriptor and optionally some
    normal bytes and then closes its connection.

    @ivar reason: The reason the connection was lost, after C{connectionLost}
        is called.
    """
    reason = None

    def __init__(self, fd, data):
        """
        @param fd: A C{int} giving a file descriptor to send over the
            connection.

        @param data: A C{str} giving data to send over the connection, or
            L{None} if no data is to be sent.
        """
        self.fd = fd
        self.data = data

    def connectionMade(self):
        """
        Send C{self.fd} and, if it is not L{None}, C{self.data}.  Then close the
        connection.
        """
        self.transport.sendFileDescriptor(self.fd)
        if self.data:
            self.transport.write(self.data)
        self.transport.loseConnection()

    def connectionLost(self, reason):
        ConnectableProtocol.connectionLost(self, reason)
        self.reason = reason