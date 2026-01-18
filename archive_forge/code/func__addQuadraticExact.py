from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import (
import math
def _addQuadraticExact(self, c0, c1, c2):
    self.value += calcQuadraticArcLengthC(c0, c1, c2)