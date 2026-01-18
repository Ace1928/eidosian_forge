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
def _buildComponents(self, componentFlags):
    if self.handleOverflowingTransforms:
        overflowing = any((s > 2 or s < -2 for glyphName, transformation in self.components for s in transformation[:4]))
    components = []
    for glyphName, transformation in self.components:
        if glyphName not in self.glyphSet:
            self.log.warning(f"skipped non-existing component '{glyphName}'")
            continue
        if self.points or (self.handleOverflowingTransforms and overflowing):
            self._decompose(glyphName, transformation)
            continue
        component = GlyphComponent()
        component.glyphName = glyphName
        component.x, component.y = (otRound(v) for v in transformation[4:])
        transformation = tuple((floatToFixedToFloat(v, 14) for v in transformation[:4]))
        if transformation != (1, 0, 0, 1):
            if self.handleOverflowingTransforms and any((MAX_F2DOT14 < s <= 2 for s in transformation)):
                transformation = tuple((MAX_F2DOT14 if MAX_F2DOT14 < s <= 2 else s for s in transformation))
            component.transform = (transformation[:2], transformation[2:])
        component.flags = componentFlags
        components.append(component)
    return components