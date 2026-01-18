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
def _moveTo(self, pt):
    contour = Contour([], [])
    self.contours.append(contour)
    contour.points.append(pt)
    contour.tags.append(FT_CURVE_TAG_ON)