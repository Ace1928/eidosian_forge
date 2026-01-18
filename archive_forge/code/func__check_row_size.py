from __future__ import division
import sys
import unicodedata
from functools import reduce
def _check_row_size(self, array):
    """Check that the specified array fits the previous rows size
        """
    if not self._row_size:
        self._row_size = len(array)
    elif self._row_size != len(array):
        raise ArraySizeError('array should contain %d elements' % self._row_size)