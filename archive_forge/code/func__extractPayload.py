import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _extractPayload(self):
    """
        Extracts payload information from C{self._remainingData}.

        Splits C{self._remainingData} at the end of the netstring.  The
        first part becomes C{self._payload}, the second part is stored
        in C{self._remainingData}.

        If the netstring is not yet complete, the whole content of
        C{self._remainingData} is moved to C{self._payload}.
        """
    if self._payloadComplete():
        remainingPayloadSize = self._expectedPayloadSize - self._currentPayloadSize
        self._payload.write(self._remainingData[:remainingPayloadSize])
        self._remainingData = self._remainingData[remainingPayloadSize:]
        self._currentPayloadSize = self._expectedPayloadSize
    else:
        self._payload.write(self._remainingData)
        self._currentPayloadSize += len(self._remainingData)
        self._remainingData = b''