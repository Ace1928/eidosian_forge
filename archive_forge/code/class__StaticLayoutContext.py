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
class _StaticLayoutContext(_LayoutContext):

    def __init__(self, layout: TextLayout, document: AbstractDocument, colors_iter: RunIterator, background_iter: AbstractRunIterator):
        super().__init__(layout, document, colors_iter, background_iter)
        self.vertex_lists = layout._vertex_lists
        self.boxes = layout._boxes

    def add_list(self, vertex_list: _LayoutVertexList):
        self.vertex_lists.append(vertex_list)