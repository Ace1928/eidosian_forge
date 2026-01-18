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
class UNIXClientEndpoint:
    """
    UnixSocket client endpoint.
    """

    def __init__(self, reactor, path, timeout=30, checkPID=0):
        """
        @param reactor: An L{IReactorUNIX} provider.

        @param path: The path to the Unix socket file, used when connecting
        @type path: str

        @param timeout: Number of seconds to wait before assuming the
            connection has failed.
        @type timeout: int

        @param checkPID: If True, check for a pid file to verify that a server
            is listening.
        @type checkPID: bool
        """
        self._reactor = reactor
        self._path = path
        self._timeout = timeout
        self._checkPID = checkPID

    def connect(self, protocolFactory):
        """
        Implement L{IStreamClientEndpoint.connect} to connect via a
        UNIX Socket
        """
        try:
            wf = _WrappingFactory(protocolFactory)
            self._reactor.connectUNIX(self._path, wf, timeout=self._timeout, checkPID=self._checkPID)
            return wf._onConnection
        except BaseException:
            return defer.fail()