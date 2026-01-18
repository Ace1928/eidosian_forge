import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
def _convert_curve(self, pt1, pt2, pt3):
    curve = (self.current_pt, pt1, pt2, pt3)
    result = curve_to_quadratic(curve, self.max_err, self.all_quadratic)
    if self.stats is not None:
        n = str(len(result) - 2)
        self.stats[n] = self.stats.get(n, 0) + 1
    if self.all_quadratic:
        self.qCurveTo(*result[1:])
    elif len(result) == 3:
        self.qCurveTo(*result[1:])
    else:
        assert len(result) == 4
        super().curveTo(*result[1:])