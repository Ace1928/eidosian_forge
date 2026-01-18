import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
class Cu2QuPointPen(BasePointToSegmentPen):
    """A filter pen to convert cubic bezier curves to quadratic b-splines
    using the FontTools PointPen protocol.

    Args:
        other_point_pen: another PointPen used to draw the transformed outline.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: reverse the winding direction of all contours.
        stats: a dictionary counting the point numbers of quadratic segments.
        all_quadratic: if True (default), only quadratic b-splines are generated.
            if False, quadratic curves or cubic curves are generated depending
            on which one is more economical.
    """
    __points_required = {'move': (1, operator.eq), 'line': (1, operator.eq), 'qcurve': (2, operator.ge), 'curve': (3, operator.eq)}

    def __init__(self, other_point_pen, max_err, reverse_direction=False, stats=None, all_quadratic=True):
        BasePointToSegmentPen.__init__(self)
        if reverse_direction:
            self.pen = ReverseContourPointPen(other_point_pen)
        else:
            self.pen = other_point_pen
        self.max_err = max_err
        self.stats = stats
        self.all_quadratic = all_quadratic

    def _flushContour(self, segments):
        assert len(segments) >= 1
        closed = segments[0][0] != 'move'
        new_segments = []
        prev_points = segments[-1][1]
        prev_on_curve = prev_points[-1][0]
        for segment_type, points in segments:
            if segment_type == 'curve':
                for sub_points in self._split_super_bezier_segments(points):
                    on_curve, smooth, name, kwargs = sub_points[-1]
                    bcp1, bcp2 = (sub_points[0][0], sub_points[1][0])
                    cubic = [prev_on_curve, bcp1, bcp2, on_curve]
                    quad = curve_to_quadratic(cubic, self.max_err, self.all_quadratic)
                    if self.stats is not None:
                        n = str(len(quad) - 2)
                        self.stats[n] = self.stats.get(n, 0) + 1
                    new_points = [(pt, False, None, {}) for pt in quad[1:-1]]
                    new_points.append((on_curve, smooth, name, kwargs))
                    if self.all_quadratic or len(new_points) == 2:
                        new_segments.append(['qcurve', new_points])
                    else:
                        new_segments.append(['curve', new_points])
                    prev_on_curve = sub_points[-1][0]
            else:
                new_segments.append([segment_type, points])
                prev_on_curve = points[-1][0]
        if closed:
            new_segments = new_segments[-1:] + new_segments[:-1]
        self._drawPoints(new_segments)

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

    def _drawPoints(self, segments):
        pen = self.pen
        pen.beginPath()
        last_offcurves = []
        points_required = self.__points_required
        for i, (segment_type, points) in enumerate(segments):
            if segment_type in points_required:
                n, op = points_required[segment_type]
                assert op(len(points), n), f'illegal {segment_type!r} segment point count: expected {n}, got {len(points)}'
                offcurves = points[:-1]
                if i == 0:
                    last_offcurves = offcurves
                else:
                    for pt, smooth, name, kwargs in offcurves:
                        pen.addPoint(pt, None, smooth, name, **kwargs)
                pt, smooth, name, kwargs = points[-1]
                if pt is None:
                    assert segment_type == 'qcurve'
                    pass
                else:
                    pen.addPoint(pt, segment_type, smooth, name, **kwargs)
            else:
                raise AssertionError('unexpected segment type: %r' % segment_type)
        for pt, smooth, name, kwargs in last_offcurves:
            pen.addPoint(pt, None, smooth, name, **kwargs)
        pen.endPath()

    def addComponent(self, baseGlyphName, transformation):
        assert self.currentPath is None
        self.pen.addComponent(baseGlyphName, transformation)