from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class GlyphClassDefinition(Statement):
    """Example: ``@UPPERCASE = [A-Z];``."""

    def __init__(self, name, glyphs, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.glyphs = glyphs

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphs.glyphSet())

    def asFea(self, indent=''):
        return '@' + self.name + ' = ' + self.glyphs.asFea() + ';'