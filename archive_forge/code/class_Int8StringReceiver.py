import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class Int8StringReceiver(IntNStringReceiver):
    """
    A receiver for int8-prefixed strings.

    An int8 string is a string prefixed by 1 byte, the 8-bit length of
    the string.

    This class publishes the same interface as NetstringReceiver.
    """
    structFormat = '!B'
    prefixLength = calcsize(structFormat)