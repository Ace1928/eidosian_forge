import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _maxLengthSize(self):
    """
        Calculate and return the string size of C{self.MAX_LENGTH}.

        @return: The size of the string representation for C{self.MAX_LENGTH}
        @rtype: C{float}
        """
    return math.ceil(math.log10(self.MAX_LENGTH)) + 1