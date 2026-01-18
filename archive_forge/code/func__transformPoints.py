from fontTools.pens.filterPen import FilterPen, FilterPointPen
def _transformPoints(self, points):
    transformPoint = self._transformPoint
    return [transformPoint(pt) for pt in points]