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
class VisibleInfoTopBottom(typing.NamedTuple):
    """Named tuple for ListBox internals."""
    trim: int
    fill: list[VisibleInfoFillItem]

    @classmethod
    def from_raw_data(cls, trim: int, fill: Iterable[tuple[Widget, Hashable, int]]) -> Self:
        """Construct from not typed data.

        Useful for overridden cases."""
        return cls(trim=trim, fill=[VisibleInfoFillItem(*item) for item in fill])