from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def ToShortASCII(self):
    """Returns the protocol buffer as an ASCII string.
    The output is short, leaving out newlines and some other niceties.
    Defers to the C++ ProtocolPrinter class in SYMBOLIC_SHORT mode.
    """
    return self._CToASCII(ProtocolMessage._SYMBOLIC_SHORT_ASCII)