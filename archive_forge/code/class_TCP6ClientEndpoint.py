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
@implementer(interfaces.IStreamClientEndpoint)
class TCP6ClientEndpoint:
    """
    TCP client endpoint with an IPv6 configuration.

    @ivar _getaddrinfo: A hook used for testing name resolution.

    @ivar _deferToThread: A hook used for testing deferToThread.

    @ivar _GAI_ADDRESS: Index of the address portion in result of
        getaddrinfo to be used.

    @ivar _GAI_ADDRESS_HOST: Index of the actual host-address in the
        5-tuple L{_GAI_ADDRESS}.
    """
    _getaddrinfo = staticmethod(socket.getaddrinfo)
    _deferToThread = staticmethod(threads.deferToThread)
    _GAI_ADDRESS = 4
    _GAI_ADDRESS_HOST = 0

    def __init__(self, reactor, host, port, timeout=30, bindAddress=None):
        """
        @param host: An IPv6 address literal or a hostname with an
            IPv6 address

        @see: L{twisted.internet.interfaces.IReactorTCP.connectTCP}
        """
        self._reactor = reactor
        self._host = host
        self._port = port
        self._timeout = timeout
        self._bindAddress = bindAddress

    def connect(self, protocolFactory):
        """
        Implement L{IStreamClientEndpoint.connect} to connect via TCP,
        once the hostname resolution is done.
        """
        if isIPv6Address(self._host):
            d = self._resolvedHostConnect(self._host, protocolFactory)
        else:
            d = self._nameResolution(self._host)
            d.addCallback(lambda result: result[0][self._GAI_ADDRESS][self._GAI_ADDRESS_HOST])
            d.addCallback(self._resolvedHostConnect, protocolFactory)
        return d

    def _nameResolution(self, host):
        """
        Resolve the hostname string into a tuple containing the host
        IPv6 address.
        """
        return self._deferToThread(self._getaddrinfo, host, 0, socket.AF_INET6)

    def _resolvedHostConnect(self, resolvedHost, protocolFactory):
        """
        Connect to the server using the resolved hostname.
        """
        try:
            wf = _WrappingFactory(protocolFactory)
            self._reactor.connectTCP(resolvedHost, self._port, wf, timeout=self._timeout, bindAddress=self._bindAddress)
            return wf._onConnection
        except BaseException:
            return defer.fail()