import os
import re
import socket
import warnings
from typing import Optional, Sequence, Type
from unicodedata import normalize
from zope.interface import directlyProvides, implementer, provider
from constantly import NamedConstant, Names
from incremental import Version
from twisted.internet import defer, error, fdesc, interfaces, threads
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.address import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, ProcessProtocol, Protocol
from twisted.internet._resolver import HostResolution
from twisted.internet.defer import Deferred
from twisted.internet.task import LoopingCall
from twisted.logger import Logger
from twisted.plugin import IPlugin, getPlugins
from twisted.python import deprecate, log
from twisted.python.compat import _matchingString, iterbytes, nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.systemd import ListenFDs
from ._idna import _idnaBytes, _idnaText
class _WrappingProtocol(Protocol):
    """
    Wrap another protocol in order to notify my user when a connection has
    been made.
    """

    def __init__(self, connectedDeferred, wrappedProtocol):
        """
        @param connectedDeferred: The L{Deferred} that will callback
            with the C{wrappedProtocol} when it is connected.

        @param wrappedProtocol: An L{IProtocol} provider that will be
            connected.
        """
        self._connectedDeferred = connectedDeferred
        self._wrappedProtocol = wrappedProtocol
        for iface in [interfaces.IHalfCloseableProtocol, interfaces.IFileDescriptorReceiver, interfaces.IHandshakeListener]:
            if iface.providedBy(self._wrappedProtocol):
                directlyProvides(self, iface)

    def logPrefix(self):
        """
        Transparently pass through the wrapped protocol's log prefix.
        """
        if interfaces.ILoggingContext.providedBy(self._wrappedProtocol):
            return self._wrappedProtocol.logPrefix()
        return self._wrappedProtocol.__class__.__name__

    def connectionMade(self):
        """
        Connect the C{self._wrappedProtocol} to our C{self.transport} and
        callback C{self._connectedDeferred} with the C{self._wrappedProtocol}
        """
        self._wrappedProtocol.makeConnection(self.transport)
        self._connectedDeferred.callback(self._wrappedProtocol)

    def dataReceived(self, data):
        """
        Proxy C{dataReceived} calls to our C{self._wrappedProtocol}
        """
        return self._wrappedProtocol.dataReceived(data)

    def fileDescriptorReceived(self, descriptor):
        """
        Proxy C{fileDescriptorReceived} calls to our C{self._wrappedProtocol}
        """
        return self._wrappedProtocol.fileDescriptorReceived(descriptor)

    def connectionLost(self, reason):
        """
        Proxy C{connectionLost} calls to our C{self._wrappedProtocol}
        """
        return self._wrappedProtocol.connectionLost(reason)

    def readConnectionLost(self):
        """
        Proxy L{IHalfCloseableProtocol.readConnectionLost} to our
        C{self._wrappedProtocol}
        """
        self._wrappedProtocol.readConnectionLost()

    def writeConnectionLost(self):
        """
        Proxy L{IHalfCloseableProtocol.writeConnectionLost} to our
        C{self._wrappedProtocol}
        """
        self._wrappedProtocol.writeConnectionLost()

    def handshakeCompleted(self):
        """
        Proxy L{interfaces.IHandshakeListener} to our
        C{self._wrappedProtocol}.
        """
        self._wrappedProtocol.handshakeCompleted()