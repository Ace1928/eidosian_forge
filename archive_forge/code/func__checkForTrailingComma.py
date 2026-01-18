import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _checkForTrailingComma(self):
    """
        Checks if the netstring has a trailing comma at the expected position.

        @raise NetstringParseError: if the last payload character is
            anything but a comma.
        """
    if self._payload.getvalue()[-1:] != b',':
        raise NetstringParseError(self._MISSING_COMMA)