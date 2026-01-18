import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
def _segments_to_quadratic(segments, max_err, stats, all_quadratic=True):
    """Return quadratic approximations of cubic segments."""
    assert all((s[0] == 'curve' for s in segments)), 'Non-cubic given to convert'
    new_points = curves_to_quadratic([s[1] for s in segments], max_err, all_quadratic)
    n = len(new_points[0])
    assert all((len(s) == n for s in new_points[1:])), 'Converted incompatibly'
    spline_length = str(n - 2)
    stats[spline_length] = stats.get(spline_length, 0) + 1
    if all_quadratic or n == 3:
        return [('qcurve', p) for p in new_points]
    else:
        return [('curve', p) for p in new_points]