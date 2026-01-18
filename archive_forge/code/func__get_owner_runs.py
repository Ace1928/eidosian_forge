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
def _get_owner_runs(self, owner_runs: runlist.RunList, glyphs: List[Union[_InlineElementBox, Glyph]], start: int, end: int) -> None:
    owner = glyphs[start].owner
    run_start = start
    for i, glyph in enumerate(glyphs[start:end]):
        if owner != glyph.owner:
            owner_runs.set_run(run_start, i + start, owner)
            owner = glyph.owner
            run_start = i + start
    owner_runs.set_run(run_start, end, owner)