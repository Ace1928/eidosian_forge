import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class NetstringParseError(ValueError):
    """
    The incoming data is not in valid Netstring format.
    """