from __future__ import annotations
import sys
from typing import List, Optional, Type, Any, Tuple, TYPE_CHECKING
from pyglet.customtypes import AnchorX, AnchorY
from pyglet.event import EventDispatcher
from pyglet.font.base import grapheme_break
from pyglet.text import runlist
from pyglet.text.document import AbstractDocument
from pyglet.text.layout.base import _is_pyglet_doc_run, _Line, _LayoutContext, _InlineElementBox, _InvalidRange, \
from pyglet.text.layout.scrolling import ScrollableTextLayoutGroup, ScrollableTextDecorationGroup
def _update_visible_lines(self) -> None:
    start = sys.maxsize
    end = 0
    for i, line in enumerate(self.lines):
        if line.y + line.descent < self._translate_y:
            start = min(start, i)
        if line.y + line.ascent > self._translate_y - self.height:
            end = max(end, i) + 1
    for i in range(self.visible_lines.start, min(start, len(self.lines))):
        self.lines[i].delete(self)
    for i in range(end, min(self.visible_lines.end, len(self.lines))):
        self.lines[i].delete(self)
    self.invalid_vertex_lines.invalidate(start, self.visible_lines.start)
    self.invalid_vertex_lines.invalidate(self.visible_lines.end, end)
    self.visible_lines.start = start
    self.visible_lines.end = end