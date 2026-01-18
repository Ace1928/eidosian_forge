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

        Parse a bytestring as a full PROXY protocol header.

        @param line: A bytestring that represents a valid HAProxy PROXY
            protocol version 2 header.
        @type line: bytes

        @return: A L{_interfaces.IProxyInfo} containing the
            parsed data.

        @raises InvalidProxyHeader: If the bytestring does not represent a
            valid PROXY header.
        