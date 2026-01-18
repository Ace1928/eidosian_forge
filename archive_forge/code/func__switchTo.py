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
def _switchTo(self, newProto, clientFactory=None):
    """
        Switch this BinaryBoxProtocol's transport to a new protocol.  You need
        to do this 'simultaneously' on both ends of a connection; the easiest
        way to do this is to use a subclass of ProtocolSwitchCommand.

        @param newProto: the new protocol instance to switch to.

        @param clientFactory: the ClientFactory to send the
            L{twisted.internet.protocol.ClientFactory.clientConnectionLost}
            notification to.
        """
    newProtoData = self.recvd
    self.recvd = ''
    self.innerProtocol = newProto
    self.innerProtocolClientFactory = clientFactory
    newProto.makeConnection(self.transport)
    if newProtoData:
        newProto.dataReceived(newProtoData)