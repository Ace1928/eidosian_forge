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
@implementer(interfaces.IStreamServerEndpoint)
class UNIXServerEndpoint:
    """
    UnixSocket server endpoint.
    """

    def __init__(self, reactor, address, backlog=50, mode=438, wantPID=0):
        """
        @param reactor: An L{IReactorUNIX} provider.
        @param address: The path to the Unix socket file, used when listening
        @param backlog: number of connections to allow in backlog.
        @param mode: mode to set on the unix socket.  This parameter is
            deprecated.  Permissions should be set on the directory which
            contains the UNIX socket.
        @param wantPID: If True, create a pidfile for the socket.
        """
        self._reactor = reactor
        self._address = address
        self._backlog = backlog
        self._mode = mode
        self._wantPID = wantPID

    def listen(self, protocolFactory):
        """
        Implement L{IStreamServerEndpoint.listen} to listen on a UNIX socket.
        """
        return defer.execute(self._reactor.listenUNIX, self._address, protocolFactory, backlog=self._backlog, mode=self._mode, wantPID=self._wantPID)