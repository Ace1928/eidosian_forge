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
def _fill_bg(self, screen: Screen, write_position: WritePosition, erase_bg: bool) -> None:
    """
        Erase/fill the background.
        (Useful for floats and when a `char` has been given.)
        """
    char: str | None
    if callable(self.char):
        char = self.char()
    else:
        char = self.char
    if erase_bg or char:
        wp = write_position
        char_obj = _CHAR_CACHE[char or ' ', '']
        for y in range(wp.ypos, wp.ypos + wp.height):
            row = screen.data_buffer[y]
            for x in range(wp.xpos, wp.xpos + wp.width):
                row[x] = char_obj