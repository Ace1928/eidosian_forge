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
def _flow_glyphs_single_line(self, glyphs: List[Union[_InlineElementBox, Glyph]], owner_runs: runlist.RunList, start: int, end: int) -> Iterator[_Line]:
    owner_iterator = owner_runs.get_run_iterator().ranges(start, end)
    font_iterator = self.document.get_font_runs(dpi=self._dpi)
    kern_iterator = runlist.FilteredRunIterator(self.document.get_style_runs('kerning'), lambda value: value is not None, 0)
    line = _Line(start)
    font = font_iterator[0]
    if self._width:
        align_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('align'), lambda value: value in ('left', 'right', 'center'), 'left')
        line.align = align_iterator[start]
    for start, end, owner in owner_iterator:
        font = font_iterator[start]
        width = 0
        owner_glyphs = []
        for kern_start, kern_end, kern in kern_iterator.ranges(start, end):
            gs = glyphs[kern_start:kern_end]
            width += sum([g.advance for g in gs])
            width += kern * (kern_end - kern_start)
            owner_glyphs.extend(zip([kern] * (kern_end - kern_start), gs))
        if owner is None:
            for kern, glyph in owner_glyphs:
                line.add_box(glyph)
        else:
            line.add_box(_GlyphBox(owner, font, owner_glyphs, width))
    if not line.boxes:
        line.ascent = font.ascent
        line.descent = font.descent
    line.paragraph_begin = line.paragraph_end = True
    yield line