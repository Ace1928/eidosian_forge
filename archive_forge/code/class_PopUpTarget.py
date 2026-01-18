from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .overlay import Overlay
from .widget import delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
class PopUpTarget(WidgetDecoration[WrappedWidget]):
    _sizing = frozenset((Sizing.BOX,))
    _selectable = True

    def __init__(self, original_widget: WrappedWidget) -> None:
        super().__init__(original_widget)
        self._pop_up = None
        self._current_widget = self._original_widget

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

    def render(self, size: tuple[int, int], focus: bool=False) -> Canvas:
        self._update_overlay(size, focus)
        return self._current_widget.render(size, focus=focus)

    def get_cursor_coords(self, size: tuple[int, int]) -> tuple[int, int] | None:
        self._update_overlay(size, True)
        return self._current_widget.get_cursor_coords(size)

    def get_pref_col(self, size: tuple[int, int]) -> int:
        self._update_overlay(size, True)
        return self._current_widget.get_pref_col(size)

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        self._update_overlay(size, True)
        return self._current_widget.keypress(size, key)

    def move_cursor_to_coords(self, size: tuple[int, int], x: int, y: int):
        self._update_overlay(size, True)
        return self._current_widget.move_cursor_to_coords(size, x, y)

    def mouse_event(self, size: tuple[int, int], event: str, button: int, col: int, row: int, focus: bool) -> bool | None:
        self._update_overlay(size, focus)
        return self._current_widget.mouse_event(size, event, button, col, row, focus)

    def pack(self, size: tuple[int, int] | None=None, focus: bool=False) -> tuple[int, int]:
        self._update_overlay(size, focus)
        return self._current_widget.pack(size)