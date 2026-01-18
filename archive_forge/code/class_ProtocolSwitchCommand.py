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
class ProtocolSwitchCommand(Command):
    """
    Use this command to switch from something Amp-derived to a different
    protocol mid-connection.  This can be useful to use amp as the
    connection-startup negotiation phase.  Since TLS is a different layer
    entirely, you can use Amp to negotiate the security parameters of your
    connection, then switch to a different protocol, and the connection will
    remain secured.
    """

    def __init__(self, _protoToSwitchToFactory, **kw):
        """
        Create a ProtocolSwitchCommand.

        @param _protoToSwitchToFactory: a ProtocolFactory which will generate
        the Protocol to switch to.

        @param kw: Keyword arguments, encoded and handled normally as
        L{Command} would.
        """
        self.protoToSwitchToFactory = _protoToSwitchToFactory
        super().__init__(**kw)

    @classmethod
    def makeResponse(cls, innerProto, proto):
        return _SwitchBox(innerProto)

    def _doCommand(self, proto):
        """
        When we emit a ProtocolSwitchCommand, lock the protocol, but don't actually
        switch to the new protocol unless an acknowledgement is received.  If
        an error is received, switch back.
        """
        d = super()._doCommand(proto)
        proto._lockForSwitch()

        def switchNow(ign):
            innerProto = self.protoToSwitchToFactory.buildProtocol(proto.transport.getPeer())
            proto._switchTo(innerProto, self.protoToSwitchToFactory)
            return ign

        def handle(ign):
            proto._unlockFromSwitch()
            self.protoToSwitchToFactory.clientConnectionFailed(None, Failure(CONNECTION_LOST))
            return ign
        return d.addCallbacks(switchNow, handle)