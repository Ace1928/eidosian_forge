from math import sqrt, degrees, atan
from fontTools.pens.basePen import BasePen, OpenContourError
from fontTools.pens.momentsPen import MomentsPen
def _zero(self):
    self.area = 0
    self.meanX = 0
    self.meanY = 0
    self.varianceX = 0
    self.varianceY = 0
    self.stddevX = 0
    self.stddevY = 0
    self.covariance = 0
    self.correlation = 0
    self.slant = 0