from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def _get_fixed_rows_sizes(self, focus: bool=False) -> tuple[Sequence[int], Sequence[int], Sequence[tuple[int] | tuple[()]]]:
    if not self.contents:
        return ((), (), ())
    widths: dict[int, int] = {}
    heights: dict[int, int] = {}
    w_h_args: dict[int, tuple[int, int] | tuple[int] | tuple[()]] = {}
    flow: list[tuple[Widget, int, bool]] = []
    box: list[int] = []
    weighted: dict[int, list[int]] = {}
    weights: list[int] = []
    weight_max_sizes: dict[int, int] = {}
    for idx, (widget, (size_kind, size_weight)) in enumerate(self.contents):
        w_sizing = widget.sizing()
        focused = focus and self.focus == widget
        if size_kind == WHSettings.PACK:
            if Sizing.FIXED in w_sizing:
                widths[idx], heights[idx] = widget.pack((), focused)
                w_h_args[idx] = ()
            if Sizing.FLOW in w_sizing:
                flow.append((widget, idx, focused))
            if not w_sizing & {Sizing.FIXED, Sizing.FLOW}:
                raise PileError(f'Unsupported sizing {w_sizing} for {size_kind.upper()}')
        elif size_kind == WHSettings.GIVEN:
            heights[idx] = size_weight
            if Sizing.BOX in w_sizing:
                box.append(idx)
            else:
                raise PileError(f'Unsupported sizing {w_sizing} for {size_kind.upper()}')
        elif size_weight <= 0:
            widths[idx] = 0
            heights[idx] = 0
            if Sizing.FLOW in w_sizing:
                w_h_args[idx] = (0,)
            else:
                w_sizing[idx] = (0, 0)
        elif Sizing.FIXED in w_sizing and w_sizing & {Sizing.BOX, Sizing.FLOW}:
            width, height = widget.pack((), focused)
            widths[idx] = width
            if Sizing.BOX in w_sizing:
                weighted.setdefault(size_weight, []).append(idx)
                weights.append(size_weight)
                weight_max_sizes.setdefault(size_weight, height)
                weight_max_sizes[size_weight] = max(weight_max_sizes[size_weight], height)
            else:
                flow.append((widget, idx, focused))
        elif Sizing.FLOW in w_sizing:
            flow.append((widget, idx, focused))
        else:
            raise PileError(f'Unsupported combination of {size_kind}, {w_sizing}')
    if not widths:
        raise PileError('No widgets providing width information')
    max_width = max(widths.values())
    for widget, idx, focused in flow:
        widths[idx] = max_width
        heights[idx] = widget.rows((max_width,), focused)
        w_h_args[idx] = (max_width,)
    if weight_max_sizes:
        max_weighted_coefficient = max((height / weight for weight, height in weight_max_sizes.items()))
        for weight in weight_max_sizes:
            height = max(int(max_weighted_coefficient * weight + 0.5), 1)
            for idx in weighted[weight]:
                heights[idx] = height
                w_h_args[idx] = (max_width, height)
    for idx in box:
        widths[idx] = max_width
        w_h_args[idx] = (max_width, heights[idx])
    return (tuple((widths[idx] for idx in range(len(widths)))), tuple((heights[idx] for idx in range(len(heights)))), tuple((w_h_args[idx] for idx in range(len(w_h_args)))))