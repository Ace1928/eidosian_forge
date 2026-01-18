import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
def _split_super_bezier_segments(self, points):
    sub_segments = []
    n = len(points) - 1
    if n == 2:
        sub_segments.append(points)
    elif n > 2:
        on_curve, smooth, name, kwargs = points[-1]
        num_sub_segments = n - 1
        for i, sub_points in enumerate(decomposeSuperBezierSegment([pt for pt, _, _, _ in points])):
            new_segment = []
            for point in sub_points[:-1]:
                new_segment.append((point, False, None, {}))
            if i == num_sub_segments - 1:
                new_segment.append((on_curve, smooth, name, kwargs))
            else:
                new_segment.append((sub_points[-1], True, None, {}))
            sub_segments.append(new_segment)
    else:
        raise AssertionError('expected 2 control points, found: %d' % n)
    return sub_segments