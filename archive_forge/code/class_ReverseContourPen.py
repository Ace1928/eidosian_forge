from fontTools.misc.arrayTools import pairwise
from fontTools.pens.filterPen import ContourFilterPen
class ReverseContourPen(ContourFilterPen):
    """Filter pen that passes outline data to another pen, but reversing
    the winding direction of all contours. Components are simply passed
    through unchanged.

    Closed contours are reversed in such a way that the first point remains
    the first point.
    """

    def __init__(self, outPen, outputImpliedClosingLine=False):
        super().__init__(outPen)
        self.outputImpliedClosingLine = outputImpliedClosingLine

    def filterContour(self, contour):
        return reversedContour(contour, self.outputImpliedClosingLine)