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
@runtime_checkable
class EstimatedSized(Protocol):
    """Widget can estimate it's size.

    PEP 424 defines API for memory-efficiency.
    For the ListBox it's a sign of the limited body length.
    The main use-case is lazy-load, where real length calculation is expensive.
    """

    def __length_hint__(self) -> int:
        ...