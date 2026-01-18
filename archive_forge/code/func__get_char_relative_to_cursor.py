from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def _get_char_relative_to_cursor(self, offset=0):
    """
        Return character relative to cursor position, or empty string
        """
    try:
        return self.text[self.cursor_position + offset]
    except IndexError:
        return ''