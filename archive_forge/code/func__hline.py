from __future__ import division
import sys
import unicodedata
from functools import reduce
def _hline(self):
    """Print an horizontal line
        """
    if not self._hline_string:
        self._hline_string = self._build_hline()
    return self._hline_string