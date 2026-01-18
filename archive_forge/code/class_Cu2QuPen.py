import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
class Cu2QuPen(FilterPen):
    """A filter pen to convert cubic bezier curves to quadratic b-splines
    using the FontTools SegmentPen protocol.

    Args:

        other_pen: another SegmentPen used to draw the transformed outline.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: flip the contours' direction but keep starting point.
        stats: a dictionary counting the point numbers of quadratic segments.
        all_quadratic: if True (default), only quadratic b-splines are generated.
            if False, quadratic curves or cubic curves are generated depending
            on which one is more economical.
    """

    def __init__(self, other_pen, max_err, reverse_direction=False, stats=None, all_quadratic=True):
        if reverse_direction:
            other_pen = ReverseContourPen(other_pen)
        super().__init__(other_pen)
        self.max_err = max_err
        self.stats = stats
        self.all_quadratic = all_quadratic

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

    def curveTo(self, *points):
        n = len(points)
        if n == 3:
            self._convert_curve(*points)
        elif n > 3:
            for segment in decomposeSuperBezierSegment(points):
                self._convert_curve(*segment)
        else:
            self.qCurveTo(*points)