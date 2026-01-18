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
@implementer(IResponderLocator)
class SimpleStringLocator:
    """
    Implement the L{AMP.locateResponder} method to do simple, string-based
    dispatch.
    """
    baseDispatchPrefix = b'amp_'

    def locateResponder(self, name):
        """
        Locate a callable to invoke when executing the named command.

        @return: a function with the name C{"amp_" + name} on the same
            instance, or None if no such function exists.
            This function will then be called with the L{AmpBox} itself as an
            argument.

        @param name: the normalized name (from the wire) of the command.
        @type name: C{bytes}
        """
        fName = nativeString(self.baseDispatchPrefix + name.upper())
        return getattr(self, fName, None)