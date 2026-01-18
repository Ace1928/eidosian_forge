from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
@runtime_checkable
class SupportsRelativeScroll(WidgetProto, Protocol):
    """Relative scroll-specific methods."""

    def require_relative_scroll(self, size: tuple[int, int], focus: bool=False) -> bool:
        ...

    def get_first_visible_pos(self, size: tuple[int, int], focus: bool=False) -> int:
        ...

    def get_visible_amount(self, size: tuple[int, int], focus: bool=False) -> int:
        ...