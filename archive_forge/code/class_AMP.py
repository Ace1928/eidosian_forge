from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
class AMP(BinaryBoxProtocol, BoxDispatcher, CommandLocator, SimpleStringLocator):
    """
    This protocol is an AMP connection.  See the module docstring for protocol
    details.
    """
    _ampInitialized = False

    def __init__(self, boxReceiver=None, locator=None):
        self._ampInitialized = True
        if boxReceiver is None:
            boxReceiver = self
        if locator is None:
            locator = self
        BoxDispatcher.__init__(self, locator)
        BinaryBoxProtocol.__init__(self, boxReceiver)

    def locateResponder(self, name):
        """
        Unify the implementations of L{CommandLocator} and
        L{SimpleStringLocator} to perform both kinds of dispatch, preferring
        L{CommandLocator}.

        @type name: C{bytes}
        """
        firstResponder = CommandLocator.locateResponder(self, name)
        if firstResponder is not None:
            return firstResponder
        secondResponder = SimpleStringLocator.locateResponder(self, name)
        return secondResponder

    def __repr__(self) -> str:
        """
        A verbose string representation which gives us information about this
        AMP connection.
        """
        if self.innerProtocol is not None:
            innerRepr = f' inner {self.innerProtocol!r}'
        else:
            innerRepr = ''
        return f'<{self.__class__.__name__}{innerRepr} at 0x{id(self):x}>'

    def makeConnection(self, transport):
        """
        Emit a helpful log message when the connection is made.
        """
        if not self._ampInitialized:
            AMP.__init__(self)
        self._transportPeer = transport.getPeer()
        self._transportHost = transport.getHost()
        log.msg('%s connection established (HOST:%s PEER:%s)' % (self.__class__.__name__, self._transportHost, self._transportPeer))
        BinaryBoxProtocol.makeConnection(self, transport)

    def connectionLost(self, reason):
        """
        Emit a helpful log message when the connection is lost.
        """
        log.msg('%s connection lost (HOST:%s PEER:%s)' % (self.__class__.__name__, self._transportHost, self._transportPeer))
        BinaryBoxProtocol.connectionLost(self, reason)
        self.transport = None