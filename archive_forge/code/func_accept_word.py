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
def accept_word(event: KeyPressEvent):
    """Fill partial autosuggestion by word"""
    buffer = event.current_buffer
    suggestion = buffer.suggestion
    if suggestion:
        t = re.split('(\\S+\\s+)', suggestion.text)
        buffer.insert_text(next((x for x in t if x), ''))
    else:
        nc.forward_word(event)