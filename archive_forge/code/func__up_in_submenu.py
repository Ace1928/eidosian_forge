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
@kb.add('up', filter=in_sub_menu)
def _up_in_submenu(event: E) -> None:
    """Select previous (enabled) menu item or return to main menu."""
    menu = self._get_menu(len(self.selected_menu) - 2)
    index = self.selected_menu[-1]
    previous_indexes = [i for i, item in enumerate(menu.children) if i < index and (not item.disabled)]
    if previous_indexes:
        self.selected_menu[-1] = previous_indexes[-1]
    elif len(self.selected_menu) == 2:
        self.selected_menu.pop()