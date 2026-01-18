from __future__ import division
import sys
import unicodedata
from functools import reduce
def _has_hlines(self):
    """Return a boolean, if hlines are required or not
        """
    return self._deco & Texttable.HLINES > 0