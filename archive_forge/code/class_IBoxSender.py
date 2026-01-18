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
class IBoxSender(Interface):
    """
    A transport which can send L{AmpBox} objects.
    """

    def sendBox(box):
        """
        Send an L{AmpBox}.

        @raise ProtocolSwitched: if the underlying protocol has been
        switched.

        @raise ConnectionLost: if the underlying connection has already been
        lost.
        """

    def unhandledError(failure):
        """
        An unhandled error occurred in response to a box.  Log it
        appropriately.

        @param failure: a L{Failure} describing the error that occurred.
        """