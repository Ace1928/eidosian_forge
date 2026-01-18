from __future__ import annotations
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import (
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.keys import Keys
from ..key_bindings import KeyBindings
from .named_commands import get_by_name
@handle('c-j')
def _newline2(event: E) -> None:
    """
        By default, handle \\n as if it were a \\r (enter).
        (It appears that some terminals send \\n instead of \\r when pressing
        enter. - at least the Linux subsystem for Windows.)
        """
    event.key_processor.feed(KeyPress(Keys.ControlM, '\r'), first=True)