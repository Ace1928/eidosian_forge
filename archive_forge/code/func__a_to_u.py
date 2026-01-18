import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _a_to_u(self, aString):
    """Decode ASCII strings to unicode if a self.encoding is specified."""
    if isinstance(aString, six.binary_type) and self.encoding:
        return aString.decode(self.encoding)
    else:
        return aString