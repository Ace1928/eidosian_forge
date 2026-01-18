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
@text_object('g', 'm')
def _gm(event: E) -> TextObject:
    """
        Like g0, but half a screenwidth to the right. (Or as much as possible.)
        """
    w = event.app.layout.current_window
    buff = event.current_buffer
    if w and w.render_info:
        width = w.render_info.window_width
        start = buff.document.get_start_of_line_position(after_whitespace=False)
        start += int(min(width / 2, len(buff.document.current_line)))
        return TextObject(start, type=TextObjectType.INCLUSIVE)
    return TextObject(0)