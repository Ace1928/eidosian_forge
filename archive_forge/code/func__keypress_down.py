from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
def _keypress_down(self, size: tuple[int, int]) -> bool | None:
    maxcol, maxrow = size
    middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
    if middle is None:
        return True
    focus_row_offset, focus_widget, focus_pos, focus_rows, cursor = middle
    _trim_bottom, fill_below = bottom
    row_offset = focus_row_offset + focus_rows
    rows = focus_rows
    pos = focus_pos
    widget = None
    for widget, pos, rows in fill_below:
        if rows and widget.selectable():
            self.change_focus((maxcol, maxrow), pos, row_offset, 'above')
            return None
        row_offset += rows
    row_offset -= 1
    self._invalidate()
    while row_offset < maxrow:
        widget, pos = self._body.get_next(pos)
        if widget is None:
            return True
        rows = widget.rows((maxcol,))
        if rows and widget.selectable():
            self.change_focus((maxcol, maxrow), pos, row_offset, 'above')
            return None
        row_offset += rows
    if not focus_widget.selectable() or focus_row_offset + focus_rows - 1 <= 0:
        if widget is None:
            self.shift_focus((maxcol, maxrow), row_offset - rows)
            return None
        self.change_focus((maxcol, maxrow), pos, row_offset - rows, 'above')
        return None
    if cursor is not None:
        _x, y = cursor
        if y + focus_row_offset - 1 < 0:
            if widget is None:
                widget, pos = self._body.get_next(pos)
                if widget is None:
                    return None
            else:
                row_offset -= rows
            if row_offset >= maxrow:
                row_offset = maxrow - 1
            self.change_focus((maxcol, maxrow), pos, row_offset, 'above')
            return None
    self.shift_focus((maxcol, maxrow), focus_row_offset - 1)
    return None