from fontTools.misc.arrayTools import updateBounds, pointInRect, unionRect
from fontTools.misc.bezierTools import calcCubicBounds, calcQuadraticBounds
from fontTools.pens.basePen import BasePen
Pen to calculate the bounds of a shape. It calculates the
    correct bounds even when the shape contains curves that don't
    have points on their extremes. This is somewhat slower to compute
    than the "control bounds".

    When the shape has been drawn, the bounds are available as the
    ``bounds`` attribute of the pen object. It's a 4-tuple::

            (xMin, yMin, xMax, yMax)
    