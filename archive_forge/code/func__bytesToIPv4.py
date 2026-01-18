import binascii
import struct
from typing import Callable, Tuple, Type, Union
from zope.interface import implementer
from constantly import ValueConstant, Values
from typing_extensions import Literal
from twisted.internet import address
from twisted.python import compat
from . import _info, _interfaces
from ._exceptions import (
@staticmethod
def _bytesToIPv4(bytestring: bytes) -> bytes:
    """
        Convert packed 32-bit IPv4 address bytes into a dotted-quad ASCII bytes
        representation of that address.

        @param bytestring: 4 octets representing an IPv4 address.
        @type bytestring: L{bytes}

        @return: a dotted-quad notation IPv4 address.
        @rtype: L{bytes}
        """
    return b'.'.join((('%i' % (ord(b),)).encode('ascii') for b in compat.iterbytes(bytestring)))