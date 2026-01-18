import os
import ctypes
import platform
import subprocess
import collections
import math
import freetype
from freetype.raw import FT_Outline_Get_Bitmap, FT_Outline_Get_BBox, FT_Outline_Get_CBox
from freetype.ft_types import FT_Pos
from freetype.ft_structs import FT_Vector, FT_BBox, FT_Bitmap, FT_Outline
from freetype.ft_enums import (
from freetype.ft_errors import FT_Exception
from fontTools.pens.basePen import BasePen, PenError
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform
def _curveToOne(self, p1, p2, p3):
    if not (self.contours and len(self.contours[-1].points) > 0):
        raise PenError('Contour missing required initial moveTo')
    t1, t2, t3 = (FT_CURVE_TAG_CUBIC, FT_CURVE_TAG_CUBIC, FT_CURVE_TAG_ON)
    contour = self.contours[-1]
    for p, t in ((p1, t1), (p2, t2), (p3, t3)):
        contour.points.append(p)
        contour.tags.append(t)