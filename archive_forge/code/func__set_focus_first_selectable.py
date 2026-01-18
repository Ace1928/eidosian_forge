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
def _set_focus_first_selectable(self, size: tuple[int, int], focus: bool) -> None:
    """Choose the first visible, selectable widget below the current focus as the focus widget."""
    maxcol, maxrow = size
    self.set_focus_valign_pending = None
    self.set_focus_pending = None
    middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus=focus)
    if middle is None:
        return
    row_offset, focus_widget, _focus_pos, focus_rows, _cursor = middle
    _trim_top, _fill_above = top
    trim_bottom, fill_below = bottom
    if focus_widget.selectable():
        return
    if trim_bottom:
        fill_below = fill_below[:-1]
    new_row_offset = row_offset + focus_rows
    for widget, pos, rows in fill_below:
        if widget.selectable():
            self._body.set_focus(pos)
            self.shift_focus((maxcol, maxrow), new_row_offset)
            return
        new_row_offset += rows