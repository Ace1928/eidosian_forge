from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
import urwid
from urwid.canvas import Canvas, CanvasJoin, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def _get_fixed_column_sizes(self, focus: bool=False) -> tuple[Sequence[int], Sequence[int], Sequence[tuple[int] | tuple[()]]]:
    """Get column widths, heights and render size parameters"""
    widths: dict[int, int] = {}
    heights: dict[int, int] = {}
    w_h_args: dict[int, tuple[int, int] | tuple[int] | tuple[()]] = {}
    box: list[int] = []
    weighted: dict[int, list[tuple[Widget, int, bool, bool]]] = {}
    weights: list[int] = []
    weight_max_sizes: dict[int, int] = {}
    for i, (widget, (size_kind, size_weight, is_box)) in enumerate(self.contents):
        w_sizing = widget.sizing()
        focused = focus and i == self.focus_position
        if size_kind == WHSettings.GIVEN:
            widths[i] = size_weight
            if is_box:
                box.append(i)
            elif Sizing.FLOW in w_sizing:
                heights[i] = widget.rows((size_weight,), focused)
                w_h_args[i] = (size_weight,)
            else:
                raise ColumnsError(f'Unsupported combination of {size_kind} box={is_box!r} for {widget}')
        elif size_kind == WHSettings.PACK and Sizing.FIXED in w_sizing and (not is_box):
            width, height = widget.pack((), focused)
            widths[i] = width
            heights[i] = height
            w_h_args[i] = ()
        elif size_weight <= 0:
            widths[i] = 0
            heights[i] = 1
            if is_box:
                box.append(i)
            else:
                w_h_args[i] = (0,)
        elif Sizing.FLOW in w_sizing or is_box:
            if Sizing.FIXED in w_sizing:
                width, height = widget.pack((), focused)
            else:
                width = self.min_width
            weighted.setdefault(size_weight, []).append((widget, i, is_box, focused))
            weights.append(size_weight)
            weight_max_sizes.setdefault(size_weight, width)
            weight_max_sizes[size_weight] = max(weight_max_sizes[size_weight], width)
        else:
            raise ColumnsError(f'Unsupported combination of {size_kind} box={is_box!r} for {widget}')
    if weight_max_sizes:
        max_weighted_coefficient = max((width / weight for weight, width in weight_max_sizes.items()))
        for weight in weight_max_sizes:
            width = max(int(max_weighted_coefficient * weight + 0.5), self.min_width)
            for widget, i, is_box, focused in weighted[weight]:
                widths[i] = width
                if not is_box:
                    heights[i] = widget.rows((width,), focused)
                    w_h_args[i] = (width,)
                else:
                    box.append(i)
    if not heights:
        raise ColumnsError(f'No height information for pack {self!r} as FIXED')
    max_height = max(heights.values())
    for idx in box:
        heights[idx] = max_height
        w_h_args[idx] = (widths[idx], max_height)
    return (tuple((widths[idx] for idx in range(len(widths)))), tuple((heights[idx] for idx in range(len(heights)))), tuple((w_h_args[idx] for idx in range(len(w_h_args)))))