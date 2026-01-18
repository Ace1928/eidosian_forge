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
class UnknownRecord(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    Encapsulate the wire data for unknown record types so that they can
    pass through the system unchanged.

    @type data: L{bytes}
    @ivar data: Wire data which makes up this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be cached.

    @since: 11.1
    """
    TYPE = None
    fancybasename = 'UNKNOWN'
    compareAttributes = ('data', 'ttl')
    showAttributes = (('data', _nicebytes), 'ttl')

    def __init__(self, data=b'', ttl=None):
        self.data = data
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        """
        Write the raw bytes corresponding to this record's payload to the
        stream.
        """
        strio.write(self.data)

    def decode(self, strio, length=None):
        """
        Load the bytes which are part of this record from the stream and store
        them unparsed and unmodified.
        """
        if length is None:
            raise Exception('must know length for unknown record types')
        self.data = readPrecisely(strio, length)

    def __hash__(self):
        return hash((self.data, self.ttl))