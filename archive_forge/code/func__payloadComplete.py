import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _payloadComplete(self):
    """
        Checks if enough data have been received to complete the netstring.

        @return: C{True} iff the received data contain at least as many
            characters as specified in the length section of the
            netstring
        @rtype: C{bool}
        """
    return len(self._remainingData) + self._currentPayloadSize >= self._expectedPayloadSize