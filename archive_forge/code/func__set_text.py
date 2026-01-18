from __future__ import unicode_literals
from .auto_suggest import AutoSuggest
from .clipboard import ClipboardData
from .completion import Completer, Completion, CompleteEvent
from .document import Document
from .enums import IncrementalSearchDirection
from .filters import to_simple_filter
from .history import History, InMemoryHistory
from .search_state import SearchState
from .selection import SelectionType, SelectionState, PasteMode
from .utils import Event
from .cache import FastDictCache
from .validation import ValidationError
from six.moves import range
import os
import re
import six
import subprocess
import tempfile
def _set_text(self, value):
    """ set text at current working_index. Return whether it changed. """
    working_index = self.working_index
    working_lines = self._working_lines
    original_value = working_lines[working_index]
    working_lines[working_index] = value
    if len(value) != len(original_value):
        return True
    elif value != original_value:
        return True
    return False