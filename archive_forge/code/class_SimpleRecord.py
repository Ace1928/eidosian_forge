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
class SimpleRecord(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    A Resource Record which consists of a single RFC 1035 domain-name.

    @type name: L{Name}
    @ivar name: The name associated with this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    showAttributes = (('name', 'name', '%s'), 'ttl')
    compareAttributes = ('name', 'ttl')
    TYPE: Optional[int] = None
    name = None

    def __init__(self, name=b'', ttl=None):
        """
        @param name: See L{SimpleRecord.name}
        @type name: L{bytes} or L{str}
        """
        self.name = Name(name)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        self.name.encode(strio, compDict)

    def decode(self, strio, length=None):
        self.name = Name()
        self.name.decode(strio)

    def __hash__(self):
        return hash(self.name)