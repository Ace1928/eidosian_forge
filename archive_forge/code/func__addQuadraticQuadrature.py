from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import (
import math
def _addQuadraticQuadrature(self, c0, c1, c2):
    self.value += approximateQuadraticArcLengthC(c0, c1, c2)