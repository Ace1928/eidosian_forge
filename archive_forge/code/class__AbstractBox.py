from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Pattern, Union, Optional, List, Any, Tuple, Callable, Iterator, Type, Dict, \
import pyglet
from pyglet import graphics
from pyglet.customtypes import AnchorX, AnchorY, ContentVAlign, HorizontalAlign
from pyglet.font.base import Font, Glyph
from pyglet.gl import GL_TRIANGLES, GL_LINES, glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, \
from pyglet.image import Texture
from pyglet.text import runlist
from pyglet.text.runlist import RunIterator, AbstractRunIterator
class _AbstractBox(ABC):
    """A box has two cases, A GlyphBox and an InlineElementBox
    """
    owner: Optional[Texture]
    ascent: float
    descent: float
    advance: float
    length: int

    def __init__(self, ascent: float, descent: float, advance: float, length: int) -> None:
        self.owner = None
        self.ascent = ascent
        self.descent = descent
        self.advance = advance
        self.length = length

    @abstractmethod
    def place(self, layout: TextLayout, i: int, x: float, y: float, z: float, line_x: float, line_y: float, rotation: float, visible: bool, anchor_x: float, anchor_y: float, context: _LayoutContext) -> None:
        ...

    @abstractmethod
    def update_translation(self, x: float, y: float, z: float) -> None:
        ...

    @abstractmethod
    def update_colors(self, colors: List[int]) -> None:
        ...

    @abstractmethod
    def update_view_translation(self, translate_x: float, translate_y: float) -> None:
        ...

    @abstractmethod
    def update_rotation(self, rotation: float) -> None:
        ...

    @abstractmethod
    def update_visibility(self, visible: bool) -> None:
        ...

    @abstractmethod
    def update_anchor(self, anchor_x: float, anchor_y: float) -> None:
        ...

    @abstractmethod
    def delete(self, layout: TextLayout) -> None:
        ...

    @abstractmethod
    def get_position_in_box(self, x: int) -> int:
        ...

    @abstractmethod
    def get_point_in_box(self, position: int) -> float:
        ...