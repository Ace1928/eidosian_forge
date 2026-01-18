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
def _keypress_up(self, size: tuple[int, int]) -> bool | None:
    maxcol, maxrow = size
    middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
    if middle is None:
        return True
    focus_row_offset, focus_widget, focus_pos, _ignore, cursor = middle
    _trim_top, fill_above = top
    row_offset = focus_row_offset
    pos = focus_pos
    widget = None
    for widget, pos, rows in fill_above:
        row_offset -= rows
        if rows and widget.selectable():
            self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
            return None
    row_offset += 1
    self._invalidate()
    while row_offset > 0:
        widget, pos = self._body.get_prev(pos)
        if widget is None:
            return True
        rows = widget.rows((maxcol,), True)
        row_offset -= rows
        if rows and widget.selectable():
            self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
            return None
    if not focus_widget.selectable() or focus_row_offset + 1 >= maxrow:
        if widget is None:
            self.shift_focus((maxcol, maxrow), row_offset)
            return None
        self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
        return None
    if cursor is not None:
        _x, y = cursor
        if y + focus_row_offset + 1 >= maxrow:
            if widget is None:
                widget, pos = self._body.get_prev(pos)
                if widget is None:
                    return None
                rows = widget.rows((maxcol,), True)
                row_offset -= rows
            if -row_offset >= rows:
                row_offset = -(rows - 1)
            self.change_focus((maxcol, maxrow), pos, row_offset, 'below')
            return None
    self.shift_focus((maxcol, maxrow), focus_row_offset + 1)
    return None