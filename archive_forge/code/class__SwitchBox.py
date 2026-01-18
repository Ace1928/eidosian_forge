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
class _SwitchBox(AmpBox):
    """
    Implementation detail of ProtocolSwitchCommand: I am an AmpBox which sets
    up state for the protocol to switch.
    """

    def __init__(self, innerProto, **kw):
        """
        Create a _SwitchBox with the protocol to switch to after being sent.

        @param innerProto: the protocol instance to switch to.
        @type innerProto: an IProtocol provider.
        """
        super().__init__(**kw)
        self.innerProto = innerProto

    def __repr__(self) -> str:
        return '_SwitchBox({!r}, **{})'.format(self.innerProto, dict.__repr__(self))

    def _sendTo(self, proto):
        """
        Send me; I am the last box on the connection.  All further traffic will be
        over the new protocol.
        """
        super()._sendTo(proto)
        proto._lockForSwitch()
        proto._switchTo(self.innerProto)