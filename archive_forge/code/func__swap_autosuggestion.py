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
def _swap_autosuggestion(buffer: Buffer, provider: NavigableAutoSuggestFromHistory, direction_method: Callable):
    """
    We skip most recent history entry (in either direction) if it equals the
    current autosuggestion because if user cycles when auto-suggestion is shown
    they most likely want something else than what was suggested (otherwise
    they would have accepted the suggestion).
    """
    suggestion = buffer.suggestion
    if not suggestion:
        return
    query = _get_query(buffer.document)
    current = query + suggestion.text
    direction_method(query=query, other_than=current, history=buffer.history)
    new_suggestion = provider.get_suggestion(buffer, buffer.document)
    buffer.suggestion = new_suggestion