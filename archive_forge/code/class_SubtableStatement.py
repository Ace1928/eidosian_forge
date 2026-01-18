from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class SubtableStatement(Statement):
    """Represents a subtable break."""

    def __init__(self, location=None):
        Statement.__init__(self, location)

    def build(self, builder):
        """Calls the builder objects's ``add_subtable_break`` callback."""
        builder.add_subtable_break(self.location)

    def asFea(self, indent=''):
        return 'subtable;'