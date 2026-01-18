from __future__ import annotations
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, Iterable, NamedTuple
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.lexers import Lexer, SimpleLexer
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.search import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.utils import get_cwidth
from .processors import (
class SearchBufferControl(BufferControl):
    """
    :class:`.BufferControl` which is used for searching another
    :class:`.BufferControl`.

    :param ignore_case: Search case insensitive.
    """

    def __init__(self, buffer: Buffer | None=None, input_processors: list[Processor] | None=None, lexer: Lexer | None=None, focus_on_click: FilterOrBool=False, key_bindings: KeyBindingsBase | None=None, ignore_case: FilterOrBool=False):
        super().__init__(buffer=buffer, input_processors=input_processors, lexer=lexer, focus_on_click=focus_on_click, key_bindings=key_bindings)
        self.searcher_search_state = SearchState(ignore_case=ignore_case)