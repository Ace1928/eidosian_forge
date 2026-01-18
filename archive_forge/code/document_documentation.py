from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData

        Create a new document, with this text inserted before the buffer.
        It keeps selection ranges and cursor position in sync.
        