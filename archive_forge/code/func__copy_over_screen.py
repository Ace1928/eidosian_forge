from __future__ import annotations
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent
from .containers import Container, ScrollOffsets
from .dimension import AnyDimension, Dimension, sum_layout_dimensions, to_dimension
from .mouse_handlers import MouseHandler, MouseHandlers
from .screen import Char, Screen, WritePosition
def _copy_over_screen(self, screen: Screen, temp_screen: Screen, write_position: WritePosition, virtual_width: int) -> None:
    """
        Copy over visible screen content and "zero width escape sequences".
        """
    ypos = write_position.ypos
    xpos = write_position.xpos
    for y in range(write_position.height):
        temp_row = temp_screen.data_buffer[y + self.vertical_scroll]
        row = screen.data_buffer[y + ypos]
        temp_zero_width_escapes = temp_screen.zero_width_escapes[y + self.vertical_scroll]
        zero_width_escapes = screen.zero_width_escapes[y + ypos]
        for x in range(virtual_width):
            row[x + xpos] = temp_row[x]
            if x in temp_zero_width_escapes:
                zero_width_escapes[x + xpos] = temp_zero_width_escapes[x]