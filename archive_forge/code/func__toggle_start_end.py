from __future__ import annotations
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer, indent, unindent
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.filters import (
from prompt_toolkit.key_binding.key_bindings import Binding
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.selection import SelectionType
from ..key_bindings import ConditionalKeyBindings, KeyBindings, KeyBindingsBase
from .named_commands import get_by_name
@handle('c-x', 'c-x')
def _toggle_start_end(event: E) -> None:
    """
        Move cursor back and forth between the start and end of the current
        line.
        """
    buffer = event.current_buffer
    if buffer.document.is_cursor_at_the_end_of_line:
        buffer.cursor_position += buffer.document.get_start_of_line_position(after_whitespace=False)
    else:
        buffer.cursor_position += buffer.document.get_end_of_line_position()