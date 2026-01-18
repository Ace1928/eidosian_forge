from array import array
from typing import Any, Callable, Dict, Optional, Tuple
from fontTools.misc.fixedTools import MAX_F2DOT14, floatToFixedToFloat
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.pointPen import AbstractPointPen
from fontTools.misc.roundTools import otRound
from fontTools.pens.basePen import LoggingPen, PenError
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import flagOnCurve, flagCubic
from fontTools.ttLib.tables._g_l_y_f import Glyph
from fontTools.ttLib.tables._g_l_y_f import GlyphComponent
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables._g_l_y_f import dropImpliedOnCurvePoints
import math
class TTGlyphPointPen(_TTGlyphBasePen, LogMixin, AbstractPointPen):
    """
    Point pen used for drawing to a TrueType glyph.

    This pen can be used to construct or modify glyphs in a TrueType format
    font. After using the pen to draw, use the ``.glyph()`` method to retrieve
    a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.
    """
    drawMethod = 'drawPoints'
    transformPen = TransformPointPen

    def init(self) -> None:
        super().init()
        self._currentContourStartIndex = None

    def _isClosed(self) -> bool:
        return self._currentContourStartIndex is None

    def beginPath(self, identifier: Optional[str]=None, **kwargs: Any) -> None:
        """
        Start a new sub path.
        """
        if not self._isClosed():
            raise PenError("Didn't close previous contour.")
        self._currentContourStartIndex = len(self.points)

    def endPath(self) -> None:
        """
        End the current sub path.
        """
        if self._isClosed():
            raise PenError('Contour is already closed.')
        if self._currentContourStartIndex == len(self.points):
            self._currentContourStartIndex = None
            return
        contourStart = self.endPts[-1] + 1 if self.endPts else 0
        self.endPts.append(len(self.points) - 1)
        self._currentContourStartIndex = None
        flags = self.types
        for i in range(contourStart, len(flags)):
            if flags[i] == 'curve':
                j = i - 1
                if j < contourStart:
                    j = len(flags) - 1
                while flags[j] == 0:
                    flags[j] = flagCubic
                    j -= 1
                flags[i] = flagOnCurve

    def addPoint(self, pt: Tuple[float, float], segmentType: Optional[str]=None, smooth: bool=False, name: Optional[str]=None, identifier: Optional[str]=None, **kwargs: Any) -> None:
        """
        Add a point to the current sub path.
        """
        if self._isClosed():
            raise PenError("Can't add a point to a closed contour.")
        if segmentType is None:
            self.types.append(0)
        elif segmentType in ('line', 'move'):
            self.types.append(flagOnCurve)
        elif segmentType == 'qcurve':
            self.types.append(flagOnCurve)
        elif segmentType == 'curve':
            self.types.append('curve')
        else:
            raise AssertionError(segmentType)
        self.points.append(pt)