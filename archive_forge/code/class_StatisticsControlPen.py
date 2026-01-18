from math import sqrt, degrees, atan
from fontTools.pens.basePen import BasePen, OpenContourError
from fontTools.pens.momentsPen import MomentsPen
class StatisticsControlPen(StatisticsBase, BasePen):
    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes, using the control polygon only.

    Note that if the glyph shape is self-intersecting, the values
    are not correct (but well-defined). Moreover, area will be
    negative if contour directions are clockwise."""

    def __init__(self, glyphset=None):
        BasePen.__init__(self, glyphset)
        StatisticsBase.__init__(self)
        self._nodes = []

    def _moveTo(self, pt):
        self._nodes.append(complex(*pt))

    def _lineTo(self, pt):
        self._nodes.append(complex(*pt))

    def _qCurveToOne(self, pt1, pt2):
        for pt in (pt1, pt2):
            self._nodes.append(complex(*pt))

    def _curveToOne(self, pt1, pt2, pt3):
        for pt in (pt1, pt2, pt3):
            self._nodes.append(complex(*pt))

    def _closePath(self):
        self._update()

    def _endPath(self):
        p0 = self._getCurrentPoint()
        if p0 != self.__startPoint:
            raise OpenContourError('Glyph statistics not defined on open contours.')

    def _update(self):
        nodes = self._nodes
        n = len(nodes)
        self.area = sum((p0.real * p1.imag - p1.real * p0.imag for p0, p1 in zip(nodes, nodes[1:] + nodes[:1]))) / 2
        sumNodes = sum(nodes)
        self.meanX = meanX = sumNodes.real / n
        self.meanY = meanY = sumNodes.imag / n
        if n > 1:
            self.varianceX = varianceX = (sum((p.real * p.real for p in nodes)) - sumNodes.real * sumNodes.real / n) / (n - 1)
            self.varianceY = varianceY = (sum((p.imag * p.imag for p in nodes)) - sumNodes.imag * sumNodes.imag / n) / (n - 1)
            self.covariance = covariance = (sum((p.real * p.imag for p in nodes)) - sumNodes.real * sumNodes.imag / n) / (n - 1)
        else:
            self.varianceX = varianceX = 0
            self.varianceY = varianceY = 0
            self.covariance = covariance = 0
        StatisticsBase._update(self)