from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ValueRecordDefinition(Statement):
    """Represents a named value record definition."""

    def __init__(self, name, value, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.value = value

    def asFea(self, indent=''):
        return 'valueRecordDef {} {};'.format(self.value.asFea(), self.name)