from __future__ import division
import sys
import unicodedata
from functools import reduce
def _has_vlines(self):
    """Return a boolean, if vlines are required or not
        """
    return self._deco & Texttable.VLINES > 0