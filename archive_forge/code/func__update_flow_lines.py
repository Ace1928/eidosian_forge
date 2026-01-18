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
def _update_flow_lines(self) -> None:
    invalid_start, invalid_end = self.invalid_lines.validate()
    if invalid_end - invalid_start <= 0:
        return
    invalid_end = self._flow_lines(self.lines, invalid_start, invalid_end)
    self.invalid_vertex_lines.invalidate(invalid_start, invalid_end)