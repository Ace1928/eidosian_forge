import re
import tokenize
from io import StringIO
from typing import Callable, List, Optional, Union, Generator, Tuple
import warnings
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.layout.processors import (
from IPython.core.getipython import get_ipython
from IPython.utils.tokenutil import generate_tokens
from .filters import pass_through
def accept_or_jump_to_end(event: KeyPressEvent):
    """Apply autosuggestion or jump to end of line."""
    buffer = event.current_buffer
    d = buffer.document
    after_cursor = d.text[d.cursor_position:]
    lines = after_cursor.split('\n')
    end_of_current_line = lines[0].strip()
    suggestion = buffer.suggestion
    if suggestion is not None and suggestion.text and (end_of_current_line == ''):
        buffer.insert_text(suggestion.text)
    else:
        nc.end_of_line(event)