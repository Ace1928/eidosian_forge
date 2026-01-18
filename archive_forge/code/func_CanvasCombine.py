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
def CanvasCombine(canvas_info: Iterable[tuple[Canvas, typing.Any, bool]]) -> CompositeCanvas:
    """Stack canvases in l vertically and return resulting canvas.

    :param canvas_info: list of (canvas, position, focus) tuples:

                        position
                            a value that widget.set_focus will accept or None if not allowed
                        focus
                            True if this canvas is the one that would be in focus if the whole widget is in focus
    """
    clist = [(CompositeCanvas(c), p, f) for c, p, f in canvas_info]
    combined_canvas = CompositeCanvas()
    shards = []
    children = []
    row = 0
    focus_index = 0
    for n, (canv, pos, focus) in enumerate(clist):
        if focus:
            focus_index = n
        children.append((0, row, canv, pos))
        shards.extend(canv.shards)
        combined_canvas.coords.update(canv.translate_coords(0, row))
        for shortcut in canv.shortcuts:
            combined_canvas.shortcuts[shortcut] = pos
        row += canv.rows()
    if focus_index:
        children = [children[focus_index]] + children[:focus_index] + children[focus_index + 1:]
    combined_canvas.shards = shards
    combined_canvas.children = children
    return combined_canvas