from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def CanvasJoin(canvas_info: Iterable[tuple[Canvas, typing.Any, bool, int]]) -> CompositeCanvas:
    """
    Join canvases in l horizontally. Return result.

    :param canvas_info: list of (canvas, position, focus, cols) tuples:

                        position
                            value that widget.set_focus will accept or None if not allowed
                        focus
                            True if this canvas is the one that would be in focus if the whole widget is in focus
                        cols
                            is the number of screen columns that this widget will require,
                            if larger than the actual canvas.cols() value then this widget
                            will be padded on the right.
    """
    l2 = []
    focus_item = 0
    maxrow = 0
    for n, (canv, pos, focus, cols) in enumerate(canvas_info):
        rows = canv.rows()
        pad_right = cols - canv.cols()
        if focus:
            focus_item = n
        maxrow = max(maxrow, rows)
        l2.append((canv, pos, pad_right, rows))
    shard_lists = []
    children = []
    joined_canvas = CompositeCanvas()
    col = 0
    for canv, pos, pad_right, rows in l2:
        composite_canvas = CompositeCanvas(canv)
        if pad_right:
            composite_canvas.pad_trim_left_right(0, pad_right)
        if rows < maxrow:
            composite_canvas.pad_trim_top_bottom(0, maxrow - rows)
        joined_canvas.coords.update(composite_canvas.translate_coords(col, 0))
        for shortcut in composite_canvas.shortcuts:
            joined_canvas.shortcuts[shortcut] = pos
        shard_lists.append(composite_canvas.shards)
        children.append((col, 0, composite_canvas, pos))
        col += composite_canvas.cols()
    if focus_item:
        children = [children[focus_item]] + children[:focus_item] + children[focus_item + 1:]
    joined_canvas.shards = shards_join(shard_lists)
    joined_canvas.children = children
    return joined_canvas