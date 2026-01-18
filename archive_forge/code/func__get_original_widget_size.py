from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
def _get_original_widget_size(self, size: tuple[int, int]) -> tuple[int] | tuple[()]:
    ow = self._original_widget
    sizing = ow.sizing()
    if Sizing.FLOW in sizing:
        return (size[0],)
    if Sizing.FIXED in sizing:
        return ()
    raise ScrollableError(f'{ow!r} sizing is not supported')