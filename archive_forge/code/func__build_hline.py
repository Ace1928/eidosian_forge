from __future__ import division
import sys
import unicodedata
from functools import reduce
def _build_hline(self, is_header=False):
    """Return a string used to separated rows or separate header from
        rows
        """
    horiz = self._char_horiz
    if is_header:
        horiz = self._char_header
    s = '%s%s%s' % (horiz, [horiz, self._char_corner][self._has_vlines()], horiz)
    l = s.join([horiz * n for n in self._width])
    if self._has_border():
        l = '%s%s%s%s%s\n' % (self._char_corner, horiz, l, horiz, self._char_corner)
    else:
        l += '\n'
    return l