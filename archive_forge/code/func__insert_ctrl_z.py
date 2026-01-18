from __future__ import annotations
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import (
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.keys import Keys
from ..key_bindings import KeyBindings
from .named_commands import get_by_name
@handle('c-z')
def _insert_ctrl_z(event: E) -> None:
    """
        By default, control-Z should literally insert Ctrl-Z.
        (Ansi Ctrl-Z, code 26 in MSDOS means End-Of-File.
        In a Python REPL for instance, it's possible to type
        Control-Z followed by enter to quit.)

        When the system bindings are loaded and suspend-to-background is
        supported, that will override this binding.
        """
    event.current_buffer.insert_text(event.data)