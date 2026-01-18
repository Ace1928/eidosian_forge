from __future__ import annotations
from typing import Any
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.enums import SYSTEM_BUFFER
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import ConditionalContainer, Container, Window
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.search import SearchDirection
@handle(Keys.Escape, '!', filter=~focused & emacs_mode, is_global=True)
def _focus_me(event: E) -> None:
    """M-'!' will focus this user control."""
    event.app.layout.focus(self.window)