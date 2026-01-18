from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class GlyphClassName(Expression):
    """A glyph class name, such as ``@FRENCH_MARKS``. This must be instantiated
    with a :class:`GlyphClassDefinition` object."""

    def __init__(self, glyphclass, location=None):
        Expression.__init__(self, location)
        assert isinstance(glyphclass, GlyphClassDefinition)
        self.glyphclass = glyphclass

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphclass.glyphSet())

    def asFea(self, indent=''):
        return '@' + self.glyphclass.name