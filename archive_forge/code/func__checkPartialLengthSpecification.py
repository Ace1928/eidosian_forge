import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _checkPartialLengthSpecification(self):
    """
        Makes sure that the received data represents a valid number.

        Checks if C{self._remainingData} represents a number smaller or
        equal to C{self.MAX_LENGTH}.

        @raise NetstringParseError: if C{self._remainingData} is no
            number or is too big (checked by L{_extractLength}).
        """
    partialLengthMatch = self._LENGTH_PREFIX.match(self._remainingData)
    if not partialLengthMatch:
        raise NetstringParseError(self._MISSING_LENGTH)
    lengthSpecification = partialLengthMatch.group(1)
    self._extractLength(lengthSpecification)