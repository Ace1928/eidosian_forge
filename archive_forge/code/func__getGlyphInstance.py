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
def _getGlyphInstance(self):
    from fontTools.varLib.iup import iup_delta
    from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
    from fontTools.varLib.models import supportScalar
    glyphSet = self.glyphSet
    glyfTable = glyphSet.glyfTable
    variations = glyphSet.gvarTable.variations[self.name]
    hMetrics = glyphSet.hMetrics
    vMetrics = glyphSet.vMetrics
    coordinates, _ = glyfTable._getCoordinatesAndControls(self.name, hMetrics, vMetrics)
    origCoords, endPts = (None, None)
    for var in variations:
        scalar = supportScalar(glyphSet.location, var.axes)
        if not scalar:
            continue
        delta = var.coordinates
        if None in delta:
            if origCoords is None:
                origCoords, control = glyfTable._getCoordinatesAndControls(self.name, hMetrics, vMetrics)
                endPts = control[1] if control[0] >= 1 else list(range(len(control[1])))
            delta = iup_delta(delta, origCoords, endPts)
        coordinates += GlyphCoordinates(delta) * scalar
    glyph = copy(glyfTable[self.name])
    width, lsb, height, tsb = _setCoordinates(glyph, coordinates, glyfTable, recalcBounds=self.recalcBounds)
    self.lsb = lsb
    self.tsb = tsb
    if glyphSet.hvarTable is None:
        self.width = width
        self.height = height
    return glyph