from __future__ import annotations
import codecs
import string
from enum import Enum
from itertools import accumulate
from typing import Callable, Iterable, Tuple, TypeVar
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer, indent, reshape_text, unindent
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.document import Document
from prompt_toolkit.filters import (
from prompt_toolkit.filters.app import (
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding.digraphs import DIGRAPHS
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.key_binding.vi_state import CharacterFind, InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.selection import PasteMode, SelectionState, SelectionType
from ..key_bindings import ConditionalKeyBindings, KeyBindings, KeyBindingsBase
from .named_commands import get_by_name
@key_bindings.add(*keys, filter=~vi_waiting_for_text_object_mode & filter & vi_navigation_mode, eager=eager)
def _move_in_navigation_mode(event: E) -> None:
    """
                    Move handler for navigation mode.
                    """
    text_object = text_object_func(event)
    event.current_buffer.cursor_position += text_object.start