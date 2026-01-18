import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
def _curves_to_quadratic(self, pointsList):
    curves = []
    for current_pt, points in zip(self.current_pts, pointsList):
        curves.append(current_pt + points)
    quadratics = curves_to_quadratic(curves, [self.max_err] * len(curves))
    pointsList = []
    for quadratic in quadratics:
        pointsList.append(quadratic[1:])
    self.qCurveTo(pointsList)