import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _extractLength(self, lengthAsString):
    """
        Attempts to extract the length information of a netstring.

        @raise NetstringParseError: if the number is bigger than
            C{self.MAX_LENGTH}.
        @param lengthAsString: A chunk of data starting with a length
            specification
        @type lengthAsString: C{bytes}
        @return: The length of the netstring
        @rtype: C{int}
        """
    self._checkStringSize(lengthAsString)
    length = int(lengthAsString)
    if length > self.MAX_LENGTH:
        raise NetstringParseError(self._TOO_LONG % (self.MAX_LENGTH,))
    return length