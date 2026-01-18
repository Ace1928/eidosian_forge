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
@handle('right', filter=vi_insert_multiple_mode)
def _right_multiple(event: E) -> None:
    """
        Move all cursors to the right.
        (But keep all cursors on the same line.)
        """
    buff = event.current_buffer
    new_positions = []
    for p in buff.multiple_cursor_positions:
        row, column = buff.document.translate_index_to_position(p)
        if column < len(buff.document.lines[row]):
            p += 1
        new_positions.append(p)
    buff.multiple_cursor_positions = new_positions
    if not buff.document.is_cursor_at_the_end_of_line:
        buff.cursor_position += 1