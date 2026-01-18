import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _processLength(self, lengthMatch):
    """
        Processes the length definition of a netstring.

        Extracts and stores in C{self._expectedPayloadSize} the number
        representing the netstring size.  Removes the prefix
        representing the length specification from
        C{self._remainingData}.

        @raise NetstringParseError: if the received netstring does not
            start with a number or the number is bigger than
            C{self.MAX_LENGTH}.
        @param lengthMatch: A regular expression match object matching
            a netstring length specification
        @type lengthMatch: C{re.Match}
        """
    endOfNumber = lengthMatch.end(1)
    startOfData = lengthMatch.end(2)
    lengthString = self._remainingData[:endOfNumber]
    self._expectedPayloadSize = self._extractLength(lengthString) + 1
    self._remainingData = self._remainingData[startOfData:]