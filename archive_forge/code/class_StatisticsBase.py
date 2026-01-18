from math import sqrt, degrees, atan
from fontTools.pens.basePen import BasePen, OpenContourError
from fontTools.pens.momentsPen import MomentsPen
class StatisticsBase:

    def __init__(self):
        self._zero()

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

    def _update(self):
        self.varianceX = abs(self.varianceX)
        self.varianceY = abs(self.varianceY)
        self.stddevX = stddevX = sqrt(self.varianceX)
        self.stddevY = stddevY = sqrt(self.varianceY)
        if stddevX * stddevY == 0:
            correlation = float('NaN')
        else:
            correlation = self.covariance / (stddevX * stddevY)
            correlation = max(-1, min(1, correlation))
        self.correlation = correlation if abs(correlation) > 0.001 else 0
        slant = self.covariance / self.varianceY if self.varianceY != 0 else float('NaN')
        self.slant = slant if abs(slant) > 0.001 else 0