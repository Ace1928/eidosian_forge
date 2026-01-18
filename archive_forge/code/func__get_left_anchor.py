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
def _get_left_anchor(self) -> float:
    """Returns the anchor for the X axis from the left."""
    width = self.width
    if self._anchor_x == 'left':
        return 0
    elif self._anchor_x == 'center':
        return -(width // 2)
    elif self._anchor_x == 'right':
        return -width
    else:
        assert False, '`anchor_x` must be either "left", "center", or "right".'