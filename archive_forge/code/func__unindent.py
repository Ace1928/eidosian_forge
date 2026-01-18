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
@handle('c-c', '<', filter=has_selection)
def _unindent(event: E) -> None:
    """
        Unindent selected text.
        """
    buffer = event.current_buffer
    from_, to = buffer.document.selection_range()
    from_, _ = buffer.document.translate_index_to_position(from_)
    to, _ = buffer.document.translate_index_to_position(to)
    unindent(buffer, from_, to + 1, count=event.arg)