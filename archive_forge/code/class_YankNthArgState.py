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
class YankNthArgState(object):
    """
    For yank-last-arg/yank-nth-arg: Keep track of where we are in the history.
    """

    def __init__(self, history_position=0, n=-1, previous_inserted_word=''):
        self.history_position = history_position
        self.previous_inserted_word = previous_inserted_word
        self.n = n

    def __repr__(self):
        return '%s(history_position=%r, n=%r, previous_inserted_word=%r)' % (self.__class__.__name__, self.history_position, self.n, self.previous_inserted_word)