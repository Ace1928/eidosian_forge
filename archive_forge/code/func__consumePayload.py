import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _consumePayload(self):
    """
        Consumes the payload portion of C{self._remainingData}.

        If the payload is complete, checks for the trailing comma and
        processes the payload. If not, raises an L{IncompleteNetstring}
        exception.

        @raise IncompleteNetstring: if the payload received so far
            contains fewer characters than expected.
        @raise NetstringParseError: if the payload does not end with a
        comma.
        """
    self._extractPayload()
    if self._currentPayloadSize < self._expectedPayloadSize:
        raise IncompleteNetstring()
    self._checkForTrailingComma()
    self._state = self._PARSING_LENGTH
    self._processPayload()