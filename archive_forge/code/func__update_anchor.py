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
def _update_anchor(self) -> None:
    self._anchor_left = self._get_left_anchor()
    self._anchor_bottom = self._get_bottom_anchor()
    anchor_left, anchor_top = (self._anchor_left, self._get_top_anchor())
    for line in self.lines[self.visible_lines.start:self.visible_lines.end]:
        anchor_x = anchor_left
        for box in line.boxes:
            box.update_anchor(anchor_x, anchor_top)
            anchor_x += box.advance