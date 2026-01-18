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