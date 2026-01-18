from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from .widget import Widget, WidgetError, WidgetWarning, delegate_to_widget_mixin
class WidgetDisable(WidgetDecoration[WrappedWidget]):
    """
    A decoration widget that disables interaction with the widget it
    wraps.  This widget always passes focus=False to the wrapped widget,
    even if it somehow does become the focus.
    """
    no_cache: typing.ClassVar[list[str]] = ['rows']
    ignore_focus = True

    def selectable(self) -> Literal[False]:
        return False

    def rows(self, size: tuple[int], focus: bool=False) -> int:
        return self._original_widget.rows(size, False)

    def sizing(self) -> frozenset[Sizing]:
        return self._original_widget.sizing()

    def pack(self, size, focus: bool=False) -> tuple[int, int]:
        return self._original_widget.pack(size, False)

    def render(self, size, focus: bool=False) -> CompositeCanvas:
        canv = self._original_widget.render(size, False)
        return CompositeCanvas(canv)