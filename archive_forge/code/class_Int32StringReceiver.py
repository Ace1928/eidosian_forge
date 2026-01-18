import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class Int32StringReceiver(IntNStringReceiver):
    """
    A receiver for int32-prefixed strings.

    An int32 string is a string prefixed by 4 bytes, the 32-bit length of
    the string encoded in network byte order.

    This class publishes the same interface as NetstringReceiver.
    """
    structFormat = '!I'
    prefixLength = calcsize(structFormat)