from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import (
import math
def _addCubicRecursive(self, c0, c1, c2, c3):
    self.value += calcCubicArcLengthC(c0, c1, c2, c3, self.tolerance)