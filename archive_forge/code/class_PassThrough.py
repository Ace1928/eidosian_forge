import ast
import re
import signal
import sys
from typing import Callable, Dict, Union
from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.filters import Condition, Filter, emacs_insert_mode, has_completions
from prompt_toolkit.filters import has_focus as has_focus_impl
from prompt_toolkit.filters import (
from prompt_toolkit.layout.layout import FocusableElement
from IPython.core.getipython import get_ipython
from IPython.core.guarded_eval import _find_dunder, BINARY_OP_DUNDERS, UNARY_OP_DUNDERS
from IPython.terminal.shortcuts import auto_suggest
from IPython.utils.decorators import undoc
class PassThrough(Filter):
    """A filter allowing to implement pass-through behaviour of keybindings.

    Prompt toolkit key processor dispatches only one event per binding match,
    which means that adding a new shortcut will suppress the old shortcut
    if the keybindings are the same (unless one is filtered out).

    To stop a shortcut binding from suppressing other shortcuts:
    - add the `pass_through` filter to list of filter, and
    - call `pass_through.reply(event)` in the shortcut handler.
    """

    def __init__(self):
        self._is_replying = False

    def reply(self, event: KeyPressEvent):
        self._is_replying = True
        try:
            event.key_processor.reset()
            event.key_processor.feed_multiple(event.key_sequence)
            event.key_processor.process_keys()
        finally:
            self._is_replying = False

    def __call__(self):
        return not self._is_replying