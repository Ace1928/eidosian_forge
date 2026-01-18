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
def _update_color(self) -> None:
    colors_iter = self._document.get_style_runs('color')
    colors = []
    for start, end, color in colors_iter.ranges(0, colors_iter.end):
        if color is None:
            color = (0, 0, 0, 255)
        colors.extend(color * (end - start))
    start = 0
    for box in self._boxes:
        box.update_colors(colors[start:start + box.length * 4])
        start += box.length * 4