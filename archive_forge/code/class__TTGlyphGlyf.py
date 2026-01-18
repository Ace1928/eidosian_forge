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
class _TTGlyphGlyf(_TTGlyph):

    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        glyph, offset = self._getGlyphAndOffset()
        with self.glyphSet.pushDepth() as depth:
            if depth:
                offset = 0
            if glyph.isVarComposite():
                self._drawVarComposite(glyph, pen, False)
                return
            glyph.draw(pen, self.glyphSet.glyfTable, offset)

    def drawPoints(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
        how that works.
        """
        glyph, offset = self._getGlyphAndOffset()
        with self.glyphSet.pushDepth() as depth:
            if depth:
                offset = 0
            if glyph.isVarComposite():
                self._drawVarComposite(glyph, pen, True)
                return
            glyph.drawPoints(pen, self.glyphSet.glyfTable, offset)

    def _drawVarComposite(self, glyph, pen, isPointPen):
        from fontTools.ttLib.tables._g_l_y_f import VarComponentFlags, VAR_COMPONENT_TRANSFORM_MAPPING
        for comp in glyph.components:
            with self.glyphSet.pushLocation(comp.location, comp.flags & VarComponentFlags.RESET_UNSPECIFIED_AXES):
                try:
                    pen.addVarComponent(comp.glyphName, comp.transform, self.glyphSet.rawLocation)
                except AttributeError:
                    t = comp.transform.toTransform()
                    if isPointPen:
                        tPen = TransformPointPen(pen, t)
                        self.glyphSet[comp.glyphName].drawPoints(tPen)
                    else:
                        tPen = TransformPen(pen, t)
                        self.glyphSet[comp.glyphName].draw(tPen)

    def _getGlyphAndOffset(self):
        if self.glyphSet.location and self.glyphSet.gvarTable is not None:
            glyph = self._getGlyphInstance()
        else:
            glyph = self.glyphSet.glyfTable[self.name]
        offset = self.lsb - glyph.xMin if hasattr(glyph, 'xMin') else 0
        return (glyph, offset)

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