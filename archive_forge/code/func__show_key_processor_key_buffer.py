from __future__ import annotations
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence, Union, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth, take_using_weights, to_int, to_str
from .controls import (
from .dimension import (
from .margins import Margin
from .mouse_handlers import MouseHandlers
from .screen import _CHAR_CACHE, Screen, WritePosition
from .utils import explode_text_fragments
def _show_key_processor_key_buffer(self, new_screen: Screen) -> None:
    """
        When the user is typing a key binding that consists of several keys,
        display the last pressed key if the user is in insert mode and the key
        is meaningful to be displayed.
        E.g. Some people want to bind 'jj' to escape in Vi insert mode. But the
             first 'j' needs to be displayed in order to get some feedback.
        """
    app = get_app()
    key_buffer = app.key_processor.key_buffer
    if key_buffer and _in_insert_mode() and (not app.is_done):
        data = key_buffer[-1].data
        if get_cwidth(data) == 1:
            cpos = new_screen.get_cursor_position(self)
            new_screen.data_buffer[cpos.y][cpos.x] = _CHAR_CACHE[data, 'class:partial-key-binding']