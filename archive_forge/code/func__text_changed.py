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
def _text_changed(self):
    self.validation_error = None
    self.validation_state = ValidationState.UNKNOWN
    self.complete_state = None
    self.yank_nth_arg_state = None
    self.document_before_paste = None
    self.selection_state = None
    self.suggestion = None
    self.preferred_column = None
    self.on_text_changed.fire()