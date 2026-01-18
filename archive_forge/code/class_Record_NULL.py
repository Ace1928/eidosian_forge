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
class Record_NULL(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    A null record.

    This is an experimental record type.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    fancybasename = 'NULL'
    showAttributes = (('payload', _nicebytes), 'ttl')
    compareAttributes = ('payload', 'ttl')
    TYPE = NULL

    def __init__(self, payload=None, ttl=None):
        self.payload = payload
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(self.payload)

    def decode(self, strio, length=None):
        self.payload = readPrecisely(strio, length)

    def __hash__(self):
        return hash(self.payload)