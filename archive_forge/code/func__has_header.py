from __future__ import division
import sys
import unicodedata
from functools import reduce
def _has_header(self):
    """Return a boolean, if header line is required or not
        """
    return self._deco & Texttable.HEADER > 0