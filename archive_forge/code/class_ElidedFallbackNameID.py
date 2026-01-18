from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ElidedFallbackNameID(Statement):
    """STAT table ElidedFallbackNameID

    Args:
        value: an int pointing to an existing name table name ID
    """

    def __init__(self, value, location=None):
        Statement.__init__(self, location)
        self.value = value
        self.location = location

    def build(self, builder):
        builder.setElidedFallbackName(self.value, self.location)

    def asFea(self, indent=''):
        return f'ElidedFallbackNameID {self.value};'