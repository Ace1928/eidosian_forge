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
class HostnameEndpoint:
    """
    A name-based endpoint that connects to the fastest amongst the resolved
    host addresses.

    @cvar _DEFAULT_ATTEMPT_DELAY: The default time to use between attempts, in
        seconds, when no C{attemptDelay} is given to
        L{HostnameEndpoint.__init__}.

    @ivar _hostText: the textual representation of the hostname passed to the
        constructor.  Used to pass to the reactor's hostname resolver.
    @type _hostText: L{unicode}

    @ivar _hostBytes: the encoded bytes-representation of the hostname passed
        to the constructor.  Used to construct the L{HostnameAddress}
        associated with this endpoint.
    @type _hostBytes: L{bytes}

    @ivar _hostStr: the native-string representation of the hostname passed to
        the constructor, used for exception construction
    @type _hostStr: native L{str}

    @ivar _badHostname: a flag - hopefully false!  - indicating that an invalid
        hostname was passed to the constructor.  This might be a textual
        hostname that isn't valid IDNA, or non-ASCII bytes.
    @type _badHostname: L{bool}
    """
    _getaddrinfo = staticmethod(socket.getaddrinfo)
    _deferToThread = staticmethod(threads.deferToThread)
    _DEFAULT_ATTEMPT_DELAY = 0.3

    def __init__(self, reactor, host, port, timeout=30, bindAddress=None, attemptDelay=None):
        """
        Create a L{HostnameEndpoint}.

        @param reactor: The reactor to use for connections and delayed calls.
        @type reactor: provider of L{IReactorTCP}, L{IReactorTime} and either
            L{IReactorPluggableNameResolver} or L{IReactorPluggableResolver}.

        @param host: A hostname to connect to.
        @type host: L{bytes} or L{unicode}

        @param port: The port number to connect to.
        @type port: L{int}

        @param timeout: For each individual connection attempt, the number of
            seconds to wait before assuming the connection has failed.
        @type timeout: L{float} or L{int}

        @param bindAddress: the local address of the network interface to make
            the connections from.
        @type bindAddress: L{bytes}

        @param attemptDelay: The number of seconds to delay between connection
            attempts.
        @type attemptDelay: L{float}

        @see: L{twisted.internet.interfaces.IReactorTCP.connectTCP}
        """
        self._reactor = reactor
        self._nameResolver = self._getNameResolverAndMaybeWarn(reactor)
        [self._badHostname, self._hostBytes, self._hostText] = self._hostAsBytesAndText(host)
        self._hostStr = self._hostBytes if bytes is str else self._hostText
        self._port = port
        self._timeout = timeout
        self._bindAddress = bindAddress
        if attemptDelay is None:
            attemptDelay = self._DEFAULT_ATTEMPT_DELAY
        self._attemptDelay = attemptDelay

    def __repr__(self) -> str:
        """
        Produce a string representation of the L{HostnameEndpoint}.

        @return: A L{str}
        """
        if self._badHostname:
            host = self._hostStr
        elif isIPv6Address(self._hostStr):
            host = f'[{self._hostStr}]'
        else:
            host = nativeString(self._hostBytes)
        return ''.join(['<HostnameEndpoint ', host, ':', str(self._port), '>'])

    def _getNameResolverAndMaybeWarn(self, reactor):
        """
        Retrieve a C{nameResolver} callable and warn the caller's
        caller that using a reactor which doesn't provide
        L{IReactorPluggableNameResolver} is deprecated.

        @param reactor: The reactor to check.

        @return: A L{IHostnameResolver} provider.
        """
        if not IReactorPluggableNameResolver.providedBy(reactor):
            warningString = deprecate.getDeprecationWarningString(reactor.__class__, Version('Twisted', 17, 5, 0), format='Passing HostnameEndpoint a reactor that does not provide IReactorPluggableNameResolver (%(fqpn)s) was deprecated in %(version)s', replacement='a reactor that provides IReactorPluggableNameResolver')
            warnings.warn(warningString, DeprecationWarning, stacklevel=3)
            return _SimpleHostnameResolver(self._fallbackNameResolution)
        return reactor.nameResolver

    @staticmethod
    def _hostAsBytesAndText(host):
        """
        For various reasons (documented in the C{@ivar}'s in the class
        docstring) we need both a textual and a binary representation of the
        hostname given to the constructor.  For compatibility and convenience,
        we accept both textual and binary representations of the hostname, save
        the form that was passed, and convert into the other form.  This is
        mostly just because L{HostnameAddress} chose somewhat poorly to define
        its attribute as bytes; hopefully we can find a compatible way to clean
        this up in the future and just operate in terms of text internally.

        @param host: A hostname to convert.
        @type host: L{bytes} or C{str}

        @return: a 3-tuple of C{(invalid, bytes, text)} where C{invalid} is a
            boolean indicating the validity of the hostname, C{bytes} is a
            binary representation of C{host}, and C{text} is a textual
            representation of C{host}.
        """
        if isinstance(host, bytes):
            if isIPAddress(host) or isIPv6Address(host):
                return (False, host, host.decode('ascii'))
            else:
                try:
                    return (False, host, _idnaText(host))
                except UnicodeError:
                    host = host.decode('charmap')
        else:
            host = normalize('NFC', host)
            if isIPAddress(host) or isIPv6Address(host):
                return (False, host.encode('ascii'), host)
            else:
                try:
                    return (False, _idnaBytes(host), host)
                except UnicodeError:
                    pass
        asciibytes = host.encode('ascii', 'backslashreplace')
        return (True, asciibytes, asciibytes.decode('ascii'))

    def connect(self, protocolFactory):
        """
        Attempts a connection to each resolved address, and returns a
        connection which is established first.

        @param protocolFactory: The protocol factory whose protocol
            will be connected.
        @type protocolFactory:
            L{IProtocolFactory<twisted.internet.interfaces.IProtocolFactory>}

        @return: A L{Deferred} that fires with the connected protocol
            or fails a connection-related error.
        """
        if self._badHostname:
            return defer.fail(ValueError(f'invalid hostname: {self._hostStr}'))
        d = Deferred()
        addresses = []

        @provider(IResolutionReceiver)
        class EndpointReceiver:

            @staticmethod
            def resolutionBegan(resolutionInProgress):
                pass

            @staticmethod
            def addressResolved(address):
                addresses.append(address)

            @staticmethod
            def resolutionComplete():
                d.callback(addresses)
        self._nameResolver.resolveHostName(EndpointReceiver, self._hostText, portNumber=self._port)
        d.addErrback(lambda ignored: defer.fail(error.DNSLookupError(f"Couldn't find the hostname '{self._hostStr}'")))

        @d.addCallback
        def resolvedAddressesToEndpoints(addresses):
            for eachAddress in addresses:
                if isinstance(eachAddress, IPv6Address):
                    yield TCP6ClientEndpoint(self._reactor, eachAddress.host, eachAddress.port, self._timeout, self._bindAddress)
                if isinstance(eachAddress, IPv4Address):
                    yield TCP4ClientEndpoint(self._reactor, eachAddress.host, eachAddress.port, self._timeout, self._bindAddress)
        d.addCallback(list)

        def _canceller(d):
            d.errback(error.ConnectingCancelledError(HostnameAddress(self._hostBytes, self._port)))

        @d.addCallback
        def startConnectionAttempts(endpoints):
            """
            Given a sequence of endpoints obtained via name resolution, start
            connecting to a new one every C{self._attemptDelay} seconds until
            one of the connections succeeds, all of them fail, or the attempt
            is cancelled.

            @param endpoints: a list of all the endpoints we might try to
                connect to, as determined by name resolution.
            @type endpoints: L{list} of L{IStreamServerEndpoint}

            @return: a Deferred that fires with the result of the
                C{endpoint.connect} method that completes the fastest, or fails
                with the first connection error it encountered if none of them
                succeed.
            @rtype: L{Deferred} failing with L{error.ConnectingCancelledError}
                or firing with L{IProtocol}
            """
            if not endpoints:
                raise error.DNSLookupError(f'no results for hostname lookup: {self._hostStr}')
            iterEndpoints = iter(endpoints)
            pending = []
            failures = []
            winner = defer.Deferred(canceller=_canceller)

            def checkDone():
                if pending or checkDone.completed or checkDone.endpointsLeft:
                    return
                winner.errback(failures.pop())
            checkDone.completed = False
            checkDone.endpointsLeft = True

            @LoopingCall
            def iterateEndpoint():
                endpoint = next(iterEndpoints, None)
                if endpoint is None:
                    checkDone.endpointsLeft = False
                    checkDone()
                    return
                eachAttempt = endpoint.connect(protocolFactory)
                pending.append(eachAttempt)

                @eachAttempt.addBoth
                def noLongerPending(result):
                    pending.remove(eachAttempt)
                    return result

                @eachAttempt.addCallback
                def succeeded(result):
                    winner.callback(result)

                @eachAttempt.addErrback
                def failed(reason):
                    failures.append(reason)
                    checkDone()
            iterateEndpoint.clock = self._reactor
            iterateEndpoint.start(self._attemptDelay)

            @winner.addBoth
            def cancelRemainingPending(result):
                checkDone.completed = True
                for remaining in pending[:]:
                    remaining.cancel()
                if iterateEndpoint.running:
                    iterateEndpoint.stop()
                return result
            return winner
        return d

    def _fallbackNameResolution(self, host, port):
        """
        Resolve the hostname string into a tuple containing the host
        address.  This is method is only used when the reactor does
        not provide L{IReactorPluggableNameResolver}.

        @param host: A unicode hostname to resolve.

        @param port: The port to include in the resolution.

        @return: A L{Deferred} that fires with L{_getaddrinfo}'s
            return value.
        """
        return self._deferToThread(self._getaddrinfo, host, port, 0, socket.SOCK_STREAM)