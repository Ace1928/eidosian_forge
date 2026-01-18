from __future__ import annotations
from functools import partial
from typing import Callable, Generic, Sequence, TypeVar
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggest, DynamicAutoSuggest
from prompt_toolkit.buffer import Buffer, BufferAcceptHandler
from prompt_toolkit.completion import Completer, DynamicCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.history import History
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import AnyDimension, to_dimension
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.margins import (
from prompt_toolkit.layout.processors import (
from prompt_toolkit.lexers import DynamicLexer, Lexer
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.validation import DynamicValidator, Validator
from .toolbars import SearchToolbar
class RadioList(_DialogList[_T]):
    """
    List of radio buttons. Only one can be checked at the same time.

    :param values: List of (value, label) tuples.
    """
    open_character = '('
    close_character = ')'
    container_style = 'class:radio-list'
    default_style = 'class:radio'
    selected_style = 'class:radio-selected'
    checked_style = 'class:radio-checked'
    multiple_selection = False

    def __init__(self, values: Sequence[tuple[_T, AnyFormattedText]], default: _T | None=None) -> None:
        if default is None:
            default_values = None
        else:
            default_values = [default]
        super().__init__(values, default_values=default_values)