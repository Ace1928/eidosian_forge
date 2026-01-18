from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .overlay import Overlay
from .widget import delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
def _update_overlay(self, size: tuple[int, int], focus: bool) -> None:
    canv = self._original_widget.render(size, focus=focus)
    self._cache_original_canvas = canv
    pop_up = canv.get_pop_up()
    if pop_up:
        left, top, (w, overlay_width, overlay_height) = pop_up
        if self._pop_up != w:
            self._pop_up = w
            self._current_widget = Overlay(w, self._original_widget, ('fixed left', left), overlay_width, ('fixed top', top), overlay_height)
        else:
            self._current_widget.set_overlay_parameters(('fixed left', left), overlay_width, ('fixed top', top), overlay_height)
    else:
        self._pop_up = None
        self._current_widget = self._original_widget