from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace
from fontTools.misc.fixedTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.transform import Transform
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.pens.recordingPen import (
def _getGlyphAndOffset(self):
    if self.glyphSet.location and self.glyphSet.gvarTable is not None:
        glyph = self._getGlyphInstance()
    else:
        glyph = self.glyphSet.glyfTable[self.name]
    offset = self.lsb - glyph.xMin if hasattr(glyph, 'xMin') else 0
    return (glyph, offset)