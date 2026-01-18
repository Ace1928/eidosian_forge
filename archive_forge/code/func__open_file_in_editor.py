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
def _open_file_in_editor(self, filename):
    """
        Call editor executable.

        Return True when we received a zero return code.
        """
    visual = os.environ.get('VISUAL')
    editor = os.environ.get('EDITOR')
    editors = [visual, editor, '/usr/bin/editor', '/usr/bin/nano', '/usr/bin/pico', '/usr/bin/vi', '/usr/bin/emacs']
    for e in editors:
        if e:
            try:
                returncode = subprocess.call([e, filename])
                return returncode == 0
            except OSError:
                pass
    return False