from __future__ import annotations
from typing import Callable, Iterable, Sequence
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text.base import OneStyleAndTextTuple, StyleAndTextTuples
from prompt_toolkit.key_binding.key_bindings import KeyBindings, KeyBindingsBase
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Shadow
from .base import Border
@kb.add('left', filter=in_sub_menu)
@kb.add('c-g', filter=in_sub_menu)
@kb.add('c-c', filter=in_sub_menu)
def _back(event: E) -> None:
    """Go back to parent menu."""
    if len(self.selected_menu) > 1:
        self.selected_menu.pop()