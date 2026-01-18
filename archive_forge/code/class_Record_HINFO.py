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
class Record_HINFO(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    Host information.

    @type cpu: L{bytes}
    @ivar cpu: Specifies the CPU type.

    @type os: L{bytes}
    @ivar os: Specifies the OS.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    TYPE = HINFO
    fancybasename = 'HINFO'
    showAttributes = (('cpu', _nicebytes), ('os', _nicebytes), 'ttl')
    compareAttributes = ('cpu', 'os', 'ttl')

    def __init__(self, cpu: bytes=b'', os: bytes=b'', ttl: Union[str, bytes, int, None]=None):
        self.cpu, self.os = (cpu, os)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(struct.pack('!B', len(self.cpu)) + self.cpu)
        strio.write(struct.pack('!B', len(self.os)) + self.os)

    def decode(self, strio, length=None):
        cpu = struct.unpack('!B', readPrecisely(strio, 1))[0]
        self.cpu = readPrecisely(strio, cpu)
        os = struct.unpack('!B', readPrecisely(strio, 1))[0]
        self.os = readPrecisely(strio, os)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Record_HINFO):
            return self.os.lower() == other.os.lower() and self.cpu.lower() == other.cpu.lower() and (self.ttl == other.ttl)
        return NotImplemented

    def __hash__(self):
        return hash((self.os.lower(), self.cpu.lower()))