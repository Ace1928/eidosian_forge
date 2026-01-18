from __future__ import annotations
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent
from .containers import Container, ScrollOffsets
from .dimension import AnyDimension, Dimension, sum_layout_dimensions, to_dimension
from .mouse_handlers import MouseHandler, MouseHandlers
from .screen import Char, Screen, WritePosition
def _copy_over_write_positions(self, screen: Screen, temp_screen: Screen, write_position: WritePosition) -> None:
    """
        Copy over window write positions.
        """
    ypos = write_position.ypos
    xpos = write_position.xpos
    for win, write_pos in temp_screen.visible_windows_to_write_positions.items():
        screen.visible_windows_to_write_positions[win] = WritePosition(xpos=write_pos.xpos + xpos, ypos=write_pos.ypos + ypos - self.vertical_scroll, height=write_pos.height, width=write_pos.width)