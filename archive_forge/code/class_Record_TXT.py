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
class Record_TXT(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    Freeform text.

    @type data: L{list} of L{bytes}
    @ivar data: Freeform text which makes up this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be cached.
    """
    TYPE = TXT
    fancybasename = 'TXT'
    showAttributes = (('data', _nicebyteslist), 'ttl')
    compareAttributes = ('data', 'ttl')

    def __init__(self, *data, **kw):
        self.data = list(data)
        self.ttl = str2time(kw.get('ttl', None))

    def encode(self, strio, compDict=None):
        for d in self.data:
            strio.write(struct.pack('!B', len(d)) + d)

    def decode(self, strio, length=None):
        soFar = 0
        self.data = []
        while soFar < length:
            L = struct.unpack('!B', readPrecisely(strio, 1))[0]
            self.data.append(readPrecisely(strio, L))
            soFar += L + 1
        if soFar != length:
            log.msg('Decoded %d bytes in %s record, but rdlength is %d' % (soFar, self.fancybasename, length))

    def __hash__(self):
        return hash(tuple(self.data))