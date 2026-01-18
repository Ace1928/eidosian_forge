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
def _initialize_groups(self) -> None:
    decoration_shader = get_default_decoration_shader()
    self.background_decoration_group = self.decoration_class(decoration_shader, order=0, parent=self._user_group)
    self.foreground_decoration_group = self.decoration_class(decoration_shader, order=2, parent=self._user_group)