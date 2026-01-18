from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
@implementer(IEncodableRecord)
class Record_A(tputil.FancyEqMixin):
    """
    An IPv4 host address.

    @type address: L{bytes}
    @ivar address: The packed network-order representation of the IPv4 address
        associated with this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    compareAttributes = ('address', 'ttl')
    TYPE = A
    address = None

    def __init__(self, address='0.0.0.0', ttl=None):
        """
        @type address: L{bytes} or L{str}
        @param address: The IPv4 address associated with this record, in
            quad-dotted notation.
        """
        if isinstance(address, bytes):
            address = address.decode('ascii')
        address = socket.inet_aton(address)
        self.address = address
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(self.address)

    def decode(self, strio, length=None):
        self.address = readPrecisely(strio, 4)

    def __hash__(self):
        return hash(self.address)

    def __str__(self) -> str:
        return f'<A address={self.dottedQuad()} ttl={self.ttl}>'
    __repr__ = __str__

    def dottedQuad(self):
        return socket.inet_ntoa(self.address)