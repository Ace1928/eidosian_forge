from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class NullGlyph(Expression):
    """The NULL glyph, used in glyph deletion substitutions."""

    def __init__(self, location=None):
        Expression.__init__(self, location)

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return ()

    def asFea(self, indent=''):
        return 'NULL'