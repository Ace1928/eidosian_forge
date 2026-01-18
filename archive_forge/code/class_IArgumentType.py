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
class IArgumentType(Interface):
    """
    An L{IArgumentType} can serialize a Python object into an AMP box and
    deserialize information from an AMP box back into a Python object.

    @since: 9.0
    """

    def fromBox(name, strings, objects, proto):
        """
        Given an argument name and an AMP box containing serialized values,
        extract one or more Python objects and add them to the C{objects}
        dictionary.

        @param name: The name associated with this argument. Most commonly
            this is the key which can be used to find a serialized value in
            C{strings}.
        @type name: C{bytes}

        @param strings: The AMP box from which to extract one or more
            values.
        @type strings: C{dict}

        @param objects: The output dictionary to populate with the value for
            this argument. The key used will be derived from C{name}. It may
            differ; in Python 3, for example, the key will be a Unicode/native
            string. See L{_wireNameToPythonIdentifier}.
        @type objects: C{dict}

        @param proto: The protocol instance which received the AMP box being
            interpreted.  Most likely this is an instance of L{AMP}, but
            this is not guaranteed.

        @return: L{None}
        """

    def toBox(name, strings, objects, proto):
        """
        Given an argument name and a dictionary containing structured Python
        objects, serialize values into one or more strings and add them to
        the C{strings} dictionary.

        @param name: The name associated with this argument. Most commonly
            this is the key in C{strings} to associate with a C{bytes} giving
            the serialized form of that object.
        @type name: C{bytes}

        @param strings: The AMP box into which to insert one or more strings.
        @type strings: C{dict}

        @param objects: The input dictionary from which to extract Python
            objects to serialize. The key used will be derived from C{name}.
            It may differ; in Python 3, for example, the key will be a
            Unicode/native string. See L{_wireNameToPythonIdentifier}.
        @type objects: C{dict}

        @param proto: The protocol instance which will send the AMP box once
            it is fully populated.  Most likely this is an instance of
            L{AMP}, but this is not guaranteed.

        @return: L{None}
        """