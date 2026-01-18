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
class Record_TSIG(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    A transaction signature, encapsulated in a RR, as described
    in U{RFC 2845 <https://tools.ietf.org/html/rfc2845>}.

    @type algorithm: L{Name}
    @ivar algorithm: The name of the signature or MAC algorithm.

    @type timeSigned: L{int}
    @ivar timeSigned: Signing time, as seconds from the POSIX epoch.

    @type fudge: L{int}
    @ivar fudge: Allowable time skew, in seconds.

    @type MAC: L{bytes}
    @ivar MAC: The message digest or signature.

    @type originalID: L{int}
    @ivar originalID: A message ID.

    @type error: L{int}
    @ivar error: An error code (extended C{RCODE}) carried
          in exceptional cases.

    @type otherData: L{bytes}
    @ivar otherData: Other data carried in exceptional cases.

    """
    fancybasename = 'TSIG'
    compareAttributes = ('algorithm', 'timeSigned', 'fudge', 'MAC', 'originalID', 'error', 'otherData', 'ttl')
    showAttributes = ['algorithm', 'timeSigned', 'MAC', 'error', 'otherData']
    TYPE = TSIG

    def __init__(self, algorithm=None, timeSigned=None, fudge=5, MAC=None, originalID=0, error=OK, otherData=b'', ttl=0):
        self.algorithm = None if algorithm is None else Name(algorithm)
        self.timeSigned = timeSigned
        self.fudge = str2time(fudge)
        self.MAC = MAC
        self.originalID = originalID
        self.error = error
        self.otherData = otherData
        self.ttl = ttl

    def encode(self, strio, compDict=None):
        self.algorithm.encode(strio, compDict)
        strio.write(struct.pack('!Q', self.timeSigned)[2:])
        strio.write(struct.pack('!HH', self.fudge, len(self.MAC)))
        strio.write(self.MAC)
        strio.write(struct.pack('!HHH', self.originalID, self.error, len(self.otherData)))
        strio.write(self.otherData)

    def decode(self, strio, length=None):
        algorithm = Name()
        algorithm.decode(strio)
        self.algorithm = algorithm
        fields = struct.unpack('!QHH', b'\x00\x00' + readPrecisely(strio, 10))
        self.timeSigned, self.fudge, macLength = fields
        self.MAC = readPrecisely(strio, macLength)
        fields = struct.unpack('!HHH', readPrecisely(strio, 6))
        self.originalID, self.error, otherLength = fields
        self.otherData = readPrecisely(strio, otherLength)

    def __hash__(self):
        return hash((self.algorithm, self.timeSigned, self.MAC, self.originalID))