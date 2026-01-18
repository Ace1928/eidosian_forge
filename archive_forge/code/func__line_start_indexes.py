from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
@property
def _line_start_indexes(self):
    """
        Array pointing to the start indexes of all the lines.
        """
    if self._cache.line_indexes is None:
        line_lengths = map(len, self.lines)
        indexes = [0]
        append = indexes.append
        pos = 0
        for line_length in line_lengths:
            pos += line_length + 1
            append(pos)
        if len(indexes) > 1:
            indexes.pop()
        self._cache.line_indexes = indexes
    return self._cache.line_indexes