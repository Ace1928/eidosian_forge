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
class ListOf(Argument):
    """
    Encode and decode lists of instances of a single other argument type.

    For example, if you want to pass::

        [3, 7, 9, 15]

    You can create an argument like this::

        ListOf(Integer())

    The serialized form of the entire list is subject to the limit imposed by
    L{MAX_VALUE_LENGTH}.  List elements are represented as 16-bit length
    prefixed strings.  The argument type passed to the L{ListOf} initializer is
    responsible for producing the serialized form of each element.

    @ivar elementType: The L{Argument} instance used to encode and decode list
        elements (note, not an arbitrary L{IArgumentType} implementation:
        arguments must be implemented using only the C{fromString} and
        C{toString} methods, not the C{fromBox} and C{toBox} methods).

    @param optional: a boolean indicating whether this argument can be
        omitted in the protocol.

    @since: 10.0
    """

    def __init__(self, elementType, optional=False):
        self.elementType = elementType
        Argument.__init__(self, optional)

    def fromString(self, inString):
        """
        Convert the serialized form of a list of instances of some type back
        into that list.
        """
        strings = []
        parser = Int16StringReceiver()
        parser.stringReceived = strings.append
        parser.dataReceived(inString)
        elementFromString = self.elementType.fromString
        return [elementFromString(string) for string in strings]

    def toString(self, inObject):
        """
        Serialize the given list of objects to a single string.
        """
        strings = []
        for obj in inObject:
            serialized = self.elementType.toString(obj)
            strings.append(pack('!H', len(serialized)))
            strings.append(serialized)
        return b''.join(strings)