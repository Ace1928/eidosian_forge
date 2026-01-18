from __future__ import annotations
from prompt_toolkit import search
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, control_is_searchable, is_searching
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from ..key_bindings import key_binding
@Condition
def _previous_buffer_is_returnable() -> bool:
    """
    True if the previously focused buffer has a return handler.
    """
    prev_control = get_app().layout.search_target_buffer_control
    return bool(prev_control and prev_control.buffer.is_returnable)