import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
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