import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
class Cu2QuMultiPen:
    """A filter multi-pen to convert cubic bezier curves to quadratic b-splines
    in a interpolation-compatible manner, using the FontTools SegmentPen protocol.

    Args:

        other_pens: list of SegmentPens used to draw the transformed outlines.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: flip the contours' direction but keep starting point.

    This pen does not follow the normal SegmentPen protocol. Instead, its
    moveTo/lineTo/qCurveTo/curveTo methods take a list of tuples that are
    arguments that would normally be passed to a SegmentPen, one item for
    each of the pens in other_pens.
    """

    def __init__(self, other_pens, max_err, reverse_direction=False):
        if reverse_direction:
            other_pens = [ReverseContourPen(pen, outputImpliedClosingLine=True) for pen in other_pens]
        self.pens = other_pens
        self.max_err = max_err
        self.start_pts = None
        self.current_pts = None

    def _check_contour_is_open(self):
        if self.current_pts is None:
            raise AssertionError('moveTo is required')

    def _check_contour_is_closed(self):
        if self.current_pts is not None:
            raise AssertionError('closePath or endPath is required')

    def _add_moveTo(self):
        if self.start_pts is not None:
            for pt, pen in zip(self.start_pts, self.pens):
                pen.moveTo(*pt)
            self.start_pts = None

    def moveTo(self, pts):
        self._check_contour_is_closed()
        self.start_pts = self.current_pts = pts
        self._add_moveTo()

    def lineTo(self, pts):
        self._check_contour_is_open()
        self._add_moveTo()
        for pt, pen in zip(pts, self.pens):
            pen.lineTo(*pt)
        self.current_pts = pts

    def qCurveTo(self, pointsList):
        self._check_contour_is_open()
        if len(pointsList[0]) == 1:
            self.lineTo([(points[0],) for points in pointsList])
            return
        self._add_moveTo()
        current_pts = []
        for points, pen in zip(pointsList, self.pens):
            pen.qCurveTo(*points)
            current_pts.append((points[-1],))
        self.current_pts = current_pts

    def _curves_to_quadratic(self, pointsList):
        curves = []
        for current_pt, points in zip(self.current_pts, pointsList):
            curves.append(current_pt + points)
        quadratics = curves_to_quadratic(curves, [self.max_err] * len(curves))
        pointsList = []
        for quadratic in quadratics:
            pointsList.append(quadratic[1:])
        self.qCurveTo(pointsList)

    def curveTo(self, pointsList):
        self._check_contour_is_open()
        self._curves_to_quadratic(pointsList)

    def closePath(self):
        self._check_contour_is_open()
        if self.start_pts is None:
            for pen in self.pens:
                pen.closePath()
        self.current_pts = self.start_pts = None

    def endPath(self):
        self._check_contour_is_open()
        if self.start_pts is None:
            for pen in self.pens:
                pen.endPath()
        self.current_pts = self.start_pts = None

    def addComponent(self, glyphName, transformations):
        self._check_contour_is_closed()
        for trans, pen in zip(transformations, self.pens):
            pen.addComponent(glyphName, trans)