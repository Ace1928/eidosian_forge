from __future__ import division
import sys
import unicodedata
from functools import reduce
def _has_border(self):
    """Return a boolean, if border is required or not
        """
    return self._deco & Texttable.BORDER > 0