from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ElidedFallbackName(Statement):
    """STAT table ElidedFallbackName

    Args:
        names: a list of :class:`STATNameStatement` objects
    """

    def __init__(self, names, location=None):
        Statement.__init__(self, location)
        self.names = names
        self.location = location

    def build(self, builder):
        builder.setElidedFallbackName(self.names, self.location)

    def asFea(self, indent=''):
        indent += SHIFT
        res = 'ElidedFallbackName { \n'
        res += ('\n' + indent).join([s.asFea(indent=indent) for s in self.names]) + '\n'
        res += '};'
        return res