from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def _find_line_start_index(self, index):
    """
        For the index of a character at a certain line, calculate the index of
        the first character on that line.

        Return (row, index) tuple.
        """
    indexes = self._line_start_indexes
    pos = bisect.bisect_right(indexes, index) - 1
    return (pos, indexes[pos])