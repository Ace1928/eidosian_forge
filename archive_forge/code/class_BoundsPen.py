from fontTools.misc.arrayTools import updateBounds, pointInRect, unionRect
from fontTools.misc.bezierTools import calcCubicBounds, calcQuadraticBounds
from fontTools.pens.basePen import BasePen
class BoundsPen(ControlBoundsPen):
    """Pen to calculate the bounds of a shape. It calculates the
    correct bounds even when the shape contains curves that don't
    have points on their extremes. This is somewhat slower to compute
    than the "control bounds".

    When the shape has been drawn, the bounds are available as the
    ``bounds`` attribute of the pen object. It's a 4-tuple::

            (xMin, yMin, xMax, yMax)
    """

    def _curveToOne(self, bcp1, bcp2, pt):
        self._addMoveTo()
        bounds = self.bounds
        bounds = updateBounds(bounds, pt)
        if not pointInRect(bcp1, bounds) or not pointInRect(bcp2, bounds):
            bounds = unionRect(bounds, calcCubicBounds(self._getCurrentPoint(), bcp1, bcp2, pt))
        self.bounds = bounds

    def _qCurveToOne(self, bcp, pt):
        self._addMoveTo()
        bounds = self.bounds
        bounds = updateBounds(bounds, pt)
        if not pointInRect(bcp, bounds):
            bounds = unionRect(bounds, calcQuadraticBounds(self._getCurrentPoint(), bcp, pt))
        self.bounds = bounds