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
def _parseTCP(factory, port, interface='', backlog=50):
    """
    Internal parser function for L{_parseServer} to convert the string
    arguments for a TCP(IPv4) stream endpoint into the structured arguments.

    @param factory: the protocol factory being parsed, or L{None}.  (This was a
        leftover argument from when this code was in C{strports}, and is now
        mostly None and unused.)

    @type factory: L{IProtocolFactory} or L{None}

    @param port: the integer port number to bind
    @type port: C{str}

    @param interface: the interface IP to listen on
    @param backlog: the length of the listen queue
    @type backlog: C{str}

    @return: a 2-tuple of (args, kwargs), describing  the parameters to
        L{IReactorTCP.listenTCP} (or, modulo argument 2, the factory, arguments
        to L{TCP4ServerEndpoint}.
    """
    return ((int(port), factory), {'interface': interface, 'backlog': int(backlog)})