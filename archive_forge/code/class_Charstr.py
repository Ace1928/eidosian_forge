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
@implementer(IEncodable)
class Charstr:

    def __init__(self, string: bytes=b''):
        if not isinstance(string, bytes):
            raise ValueError(f'{string!r} is not a byte string')
        self.string = string

    def encode(self, strio, compDict=None):
        """
        Encode this Character string into the appropriate byte format.

        @type strio: file
        @param strio: The byte representation of this Charstr will be written
            to this file.
        """
        string = self.string
        ind = len(string)
        strio.write(_ord2bytes(ind))
        strio.write(string)

    def decode(self, strio, length=None):
        """
        Decode a byte string into this Charstr.

        @type strio: file
        @param strio: Bytes will be read from this file until the full string
            is decoded.

        @raise EOFError: Raised when there are not enough bytes available from
            C{strio}.
        """
        self.string = b''
        l = ord(readPrecisely(strio, 1))
        self.string = readPrecisely(strio, l)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Charstr):
            return self.string == other.string
        return NotImplemented

    def __hash__(self):
        return hash(self.string)

    def __str__(self) -> str:
        """
        Represent this L{Charstr} instance by its string value.
        """
        return nativeString(self.string)