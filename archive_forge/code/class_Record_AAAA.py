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
class Record_AAAA(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    An IPv6 host address.

    @type address: L{bytes}
    @ivar address: The packed network-order representation of the IPv6 address
        associated with this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.

    @see: U{http://www.faqs.org/rfcs/rfc1886.html}
    """
    TYPE = AAAA
    fancybasename = 'AAAA'
    showAttributes = (('_address', 'address', '%s'), 'ttl')
    compareAttributes = ('address', 'ttl')

    @property
    def _address(self):
        return socket.inet_ntop(AF_INET6, self.address)

    def __init__(self, address='::', ttl=None):
        """
        @type address: L{bytes} or L{str}
        @param address: The IPv6 address for this host, in RFC 2373 format.
        """
        if isinstance(address, bytes):
            address = address.decode('idna')
        self.address = socket.inet_pton(AF_INET6, address)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(self.address)

    def decode(self, strio, length=None):
        self.address = readPrecisely(strio, 16)

    def __hash__(self):
        return hash(self.address)