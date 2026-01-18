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
def _flow_lines(self, lines: List[_Line], start: int, end: int) -> int:
    margin_top_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('margin_top'), lambda value: value is not None, 0)
    margin_bottom_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('margin_bottom'), lambda value: value is not None, 0)
    line_spacing_iterator = self._document.get_style_runs('line_spacing')
    leading_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('leading'), lambda value: value is not None, 0)
    if start == 0:
        y = 0
    else:
        line = lines[start - 1]
        line_spacing = self.parse_distance(line_spacing_iterator[line.start])
        leading = self.parse_distance(leading_iterator[line.start])
        y = line.y
        if line_spacing is None:
            y += line.descent
        if line.paragraph_end:
            y -= self.parse_distance(margin_bottom_iterator[line.start])
    line_index = start
    for line in lines[start:]:
        if line.paragraph_begin:
            y -= self.parse_distance(margin_top_iterator[line.start])
            line_spacing = self.parse_distance(line_spacing_iterator[line.start])
            leading = self.parse_distance(leading_iterator[line.start])
        else:
            y -= leading
        if line_spacing is None:
            y -= line.ascent
        else:
            y -= line_spacing
        if line.align == 'left' or line.width > self.width:
            line.x = line.margin_left
        elif line.align == 'center':
            line.x = (self.width - line.margin_left - line.margin_right - line.width) // 2 + line.margin_left
        elif line.align == 'right':
            line.x = self.width - line.margin_right - line.width
        self._content_width = max(self._content_width, line.width + line.margin_left)
        if line.y == y and line_index >= end:
            break
        line.y = y
        if line_spacing is None:
            y += line.descent
        if line.paragraph_end:
            y -= self.parse_distance(margin_bottom_iterator[line.start])
        line_index += 1
    else:
        self._content_height = -y
    return line_index