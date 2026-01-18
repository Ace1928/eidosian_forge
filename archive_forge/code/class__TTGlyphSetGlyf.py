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
class _TTGlyphSetGlyf(_TTGlyphSet):

    def __init__(self, font, location, recalcBounds=True):
        self.glyfTable = font['glyf']
        super().__init__(font, location, self.glyfTable, recalcBounds=recalcBounds)
        self.gvarTable = font.get('gvar')

    def __getitem__(self, glyphName):
        return _TTGlyphGlyf(self, glyphName, recalcBounds=self.recalcBounds)