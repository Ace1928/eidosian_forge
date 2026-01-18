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
def _update_scissor_area(self) -> None:
    area = (self.left, self.bottom, self._width, self._height)
    for group in self.group_cache.values():
        group.scissor_area = area
    self.background_decoration_group.scissor_area = area
    self.foreground_decoration_group.scissor_area = area