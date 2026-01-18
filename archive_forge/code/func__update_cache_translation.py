from __future__ import annotations
import typing
from urwid import text_layout
from urwid.canvas import apply_text_layout
from urwid.split_repr import remove_defaults
from urwid.str_util import calc_width
from urwid.util import decompose_tagmarkup, get_encoding
from .constants import Align, Sizing, WrapMode
from .widget import Widget, WidgetError
def _update_cache_translation(self, maxcol: int, ta: tuple[str | bytes, list[tuple[Hashable, int]]] | None) -> None:
    if ta:
        text, _attr = ta
    else:
        text, _attr = self.get_text()
    self._cache_maxcol = maxcol
    self._cache_translation = self.layout.layout(text, maxcol, self._align_mode, self._wrap_mode)