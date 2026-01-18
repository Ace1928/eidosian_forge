from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import solveQuadratic, solveCubic
def _qCurveToOne_unfinished(self, bcp, point):
    x, y = self.testPoint
    x1, y1 = self._getCurrentPoint()
    x2, y2 = bcp
    x3, y3 = point
    c = y1
    b = (y2 - c) * 2.0
    a = y3 - c - b
    solutions = sorted(solveQuadratic(a, b, c - y))
    solutions = [t for t in solutions if ZERO_MINUS_EPSILON <= t <= ONE_PLUS_EPSILON]
    if not solutions:
        return