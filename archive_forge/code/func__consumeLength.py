import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _consumeLength(self):
    """
        Consumes the length portion of C{self._remainingData}.

        @raise IncompleteNetstring: if C{self._remainingData} contains
            a partial length specification (digits without trailing
            comma).
        @raise NetstringParseError: if the received data do not form a valid
            netstring.
        """
    lengthMatch = self._LENGTH.match(self._remainingData)
    if not lengthMatch:
        self._checkPartialLengthSpecification()
        raise IncompleteNetstring()
    self._processLength(lengthMatch)