from fontTools.pens.basePen import BasePen, OpenContourError
def _endPath(self):
    p0 = self._getCurrentPoint()
    if p0 != self.__startPoint:
        raise OpenContourError('Glyph statistics not defined on open contours.')