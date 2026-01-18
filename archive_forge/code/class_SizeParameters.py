from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class SizeParameters(Statement):
    """A ``parameters`` statement."""

    def __init__(self, DesignSize, SubfamilyID, RangeStart, RangeEnd, location=None):
        Statement.__init__(self, location)
        self.DesignSize = DesignSize
        self.SubfamilyID = SubfamilyID
        self.RangeStart = RangeStart
        self.RangeEnd = RangeEnd

    def build(self, builder):
        """Calls the builder object's ``set_size_parameters`` callback."""
        builder.set_size_parameters(self.location, self.DesignSize, self.SubfamilyID, self.RangeStart, self.RangeEnd)

    def asFea(self, indent=''):
        res = 'parameters {:.1f} {}'.format(self.DesignSize, self.SubfamilyID)
        if self.RangeStart != 0 or self.RangeEnd != 0:
            res += ' {} {}'.format(int(self.RangeStart * 10), int(self.RangeEnd * 10))
        return res + ';'