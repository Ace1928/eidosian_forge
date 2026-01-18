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
def _set_history_search(self):
    """ Set `history_search_text`. """
    if self.enable_history_search():
        if self.history_search_text is None:
            self.history_search_text = self.text
    else:
        self.history_search_text = None