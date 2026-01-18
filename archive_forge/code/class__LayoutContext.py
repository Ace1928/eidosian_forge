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
class _LayoutContext:

    def __init__(self, layout: TextLayout, document: AbstractDocument, colors_iter: RunIterator, background_iter: AbstractRunIterator) -> None:
        self.layout = layout
        self.colors_iter = colors_iter
        underline_iter = document.get_style_runs('underline')
        self.decoration_iter = runlist.ZipRunIterator((background_iter, underline_iter))
        self.baseline_iter = runlist.FilteredRunIterator(document.get_style_runs('baseline'), lambda value: value is not None, 0)

    def add_list(self, vertex_list: VertexList):
        raise NotImplementedError('abstract')

    def add_box(self, box: _AbstractBox):
        raise NotImplementedError('abstract')