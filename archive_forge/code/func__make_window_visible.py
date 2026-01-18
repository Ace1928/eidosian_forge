from __future__ import annotations
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent
from .containers import Container, ScrollOffsets
from .dimension import AnyDimension, Dimension, sum_layout_dimensions, to_dimension
from .mouse_handlers import MouseHandler, MouseHandlers
from .screen import Char, Screen, WritePosition
def _make_window_visible(self, visible_height: int, virtual_height: int, visible_win_write_pos: WritePosition, cursor_position: Point | None) -> None:
    """
        Scroll the scrollable pane, so that this window becomes visible.

        :param visible_height: Height of this `ScrollablePane` that is rendered.
        :param virtual_height: Height of the virtual, temp screen.
        :param visible_win_write_pos: `WritePosition` of the nested window on the
            temp screen.
        :param cursor_position: The location of the cursor position of this
            window on the temp screen.
        """
    min_scroll = 0
    max_scroll = virtual_height - visible_height
    if self.keep_cursor_visible():
        if cursor_position is not None:
            offsets = self.scroll_offsets
            cpos_min_scroll = cursor_position.y - visible_height + 1 + offsets.bottom
            cpos_max_scroll = cursor_position.y - offsets.top
            min_scroll = max(min_scroll, cpos_min_scroll)
            max_scroll = max(0, min(max_scroll, cpos_max_scroll))
    if self.keep_focused_window_visible():
        if visible_win_write_pos.height <= visible_height:
            window_min_scroll = visible_win_write_pos.ypos + visible_win_write_pos.height - visible_height
            window_max_scroll = visible_win_write_pos.ypos
        else:
            window_min_scroll = visible_win_write_pos.ypos
            window_max_scroll = visible_win_write_pos.ypos + visible_win_write_pos.height - visible_height
        min_scroll = max(min_scroll, window_min_scroll)
        max_scroll = min(max_scroll, window_max_scroll)
    if min_scroll > max_scroll:
        min_scroll = max_scroll
    if self.vertical_scroll > max_scroll:
        self.vertical_scroll = max_scroll
    if self.vertical_scroll < min_scroll:
        self.vertical_scroll = min_scroll