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
@implementer(IHostnameResolver)
class _SimpleHostnameResolver:
    """
    An L{IHostnameResolver} provider that invokes a provided callable
    to resolve hostnames.

    @ivar _nameResolution: the callable L{resolveHostName} invokes to
        resolve hostnames.
    @type _nameResolution: A L{callable} that accepts two arguments:
        the host to resolve and the port number to include in the
        result.
    """
    _log = Logger()

    def __init__(self, nameResolution):
        """
        Create a L{_SimpleHostnameResolver} instance.
        """
        self._nameResolution = nameResolution

    def resolveHostName(self, resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IHostResolution:
        """
        Initiate a hostname resolution.

        @param resolutionReceiver: an object that will receive each resolved
            address as it arrives.
        @type resolutionReceiver: L{IResolutionReceiver}

        @param hostName: see interface

        @param portNumber: see interface

        @param addressTypes: Ignored in this implementation.

        @param transportSemantics: Ignored in this implementation.

        @return: The resolution in progress.
        @rtype: L{IResolutionReceiver}
        """
        resolution = HostResolution(hostName)
        resolutionReceiver.resolutionBegan(resolution)
        d = self._nameResolution(hostName, portNumber)

        def cbDeliver(gairesult):
            for family, socktype, proto, canonname, sockaddr in gairesult:
                if family == socket.AF_INET6:
                    resolutionReceiver.addressResolved(IPv6Address('TCP', *sockaddr))
                elif family == socket.AF_INET:
                    resolutionReceiver.addressResolved(IPv4Address('TCP', *sockaddr))

        def ebLog(error):
            self._log.failure('while looking up {name} with {callable}', error, name=hostName, callable=self._nameResolution)
        d.addCallback(cbDeliver)
        d.addErrback(ebLog)
        d.addBoth(lambda ignored: resolutionReceiver.resolutionComplete())
        return resolution