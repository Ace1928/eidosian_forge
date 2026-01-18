from __future__ import annotations
from prompt_toolkit import search
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, control_is_searchable, is_searching
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from ..key_bindings import key_binding
@key_binding(filter=is_searching)
def abort_search(event: E) -> None:
    """
    Abort an incremental search and restore the original
    line.
    (Usually bound to ControlG/ControlC.)
    """
    search.stop_search()