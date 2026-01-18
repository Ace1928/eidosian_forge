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
class _Line:
    boxes: List[_AbstractBox]
    vertex_lists: List[VertexList]
    start: int
    align: HorizontalAlign = 'left'
    margin_left: int = 0
    margin_right: int = 0
    length: int = 0
    ascent: float = 0
    descent: float = 0
    width: float = 0
    paragraph_begin: bool = False
    paragraph_end: bool = False
    x: int
    y: int

    def __init__(self, start: int) -> None:
        self.start = start
        self.x = 0
        self.y = 0
        self.vertex_lists = []
        self.boxes = []

    def __repr__(self) -> str:
        return f'_Line({self.boxes})'

    def add_box(self, box: _AbstractBox) -> None:
        self.boxes.append(box)
        self.length += box.length
        self.ascent = max(self.ascent, box.ascent)
        self.descent = min(self.descent, box.descent)
        self.width += box.advance

    def delete(self, layout: TextLayout) -> None:
        for box in self.boxes:
            box.delete(layout)
        self.vertex_lists.clear()