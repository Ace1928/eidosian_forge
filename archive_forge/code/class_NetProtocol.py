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
class NetProtocol(Values):
    """
    Values for 'protocol' field.
    """
    UNSPEC = ValueConstant(0)
    STREAM = ValueConstant(1)
    DGRAM = ValueConstant(2)