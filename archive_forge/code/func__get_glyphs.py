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
def _get_glyphs(self) -> List[Union[_InlineElementBox, Glyph]]:
    glyphs = []
    runs = runlist.ZipRunIterator((self._document.get_font_runs(dpi=self._dpi), self._document.get_element_runs()))
    text = self._document.text
    for start, end, (font, element) in runs.ranges(0, len(text)):
        if element:
            glyphs.append(_InlineElementBox(element))
        else:
            glyphs.extend(font.get_glyphs(text[start:end]))
    return glyphs