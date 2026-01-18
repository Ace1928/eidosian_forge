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
class _WrappingFactory(ClientFactory):
    """
    Wrap a factory in order to wrap the protocols it builds.

    @ivar _wrappedFactory: A provider of I{IProtocolFactory} whose buildProtocol
        method will be called and whose resulting protocol will be wrapped.

    @ivar _onConnection: A L{Deferred} that fires when the protocol is
        connected

    @ivar _connector: A L{connector <twisted.internet.interfaces.IConnector>}
        that is managing the current or previous connection attempt.
    """
    protocol = _WrappingProtocol

    def __init__(self, wrappedFactory):
        """
        @param wrappedFactory: A provider of I{IProtocolFactory} whose
            buildProtocol method will be called and whose resulting protocol
            will be wrapped.
        """
        self._wrappedFactory = wrappedFactory
        self._onConnection = defer.Deferred(canceller=self._canceller)

    def startedConnecting(self, connector):
        """
        A connection attempt was started.  Remember the connector which started
        said attempt, for use later.
        """
        self._connector = connector

    def _canceller(self, deferred):
        """
        The outgoing connection attempt was cancelled.  Fail that L{Deferred}
        with an L{error.ConnectingCancelledError}.

        @param deferred: The L{Deferred <defer.Deferred>} that was cancelled;
            should be the same as C{self._onConnection}.
        @type deferred: L{Deferred <defer.Deferred>}

        @note: This relies on startedConnecting having been called, so it may
            seem as though there's a race condition where C{_connector} may not
            have been set.  However, using public APIs, this condition is
            impossible to catch, because a connection API
            (C{connectTCP}/C{SSL}/C{UNIX}) is always invoked before a
            L{_WrappingFactory}'s L{Deferred <defer.Deferred>} is returned to
            C{connect()}'s caller.

        @return: L{None}
        """
        deferred.errback(error.ConnectingCancelledError(self._connector.getDestination()))
        self._connector.stopConnecting()

    def doStart(self):
        """
        Start notifications are passed straight through to the wrapped factory.
        """
        self._wrappedFactory.doStart()

    def doStop(self):
        """
        Stop notifications are passed straight through to the wrapped factory.
        """
        self._wrappedFactory.doStop()

    def buildProtocol(self, addr):
        """
        Proxy C{buildProtocol} to our C{self._wrappedFactory} or errback the
        C{self._onConnection} L{Deferred} if the wrapped factory raises an
        exception or returns L{None}.

        @return: An instance of L{_WrappingProtocol} or L{None}
        """
        try:
            proto = self._wrappedFactory.buildProtocol(addr)
            if proto is None:
                raise error.NoProtocol()
        except BaseException:
            self._onConnection.errback()
        else:
            return self.protocol(self._onConnection, proto)

    def clientConnectionFailed(self, connector, reason):
        """
        Errback the C{self._onConnection} L{Deferred} when the
        client connection fails.
        """
        if not self._onConnection.called:
            self._onConnection.errback(reason)