from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class GlyphClassDefStatement(Statement):
    """Example: ``GlyphClassDef @UPPERCASE, [B], [C], [D];``. The parameters
    must be either :class:`GlyphClass` or :class:`GlyphClassName` objects, or
    ``None``."""

    def __init__(self, baseGlyphs, markGlyphs, ligatureGlyphs, componentGlyphs, location=None):
        Statement.__init__(self, location)
        self.baseGlyphs, self.markGlyphs = (baseGlyphs, markGlyphs)
        self.ligatureGlyphs = ligatureGlyphs
        self.componentGlyphs = componentGlyphs

    def build(self, builder):
        """Calls the builder's ``add_glyphClassDef`` callback."""
        base = self.baseGlyphs.glyphSet() if self.baseGlyphs else tuple()
        liga = self.ligatureGlyphs.glyphSet() if self.ligatureGlyphs else tuple()
        mark = self.markGlyphs.glyphSet() if self.markGlyphs else tuple()
        comp = self.componentGlyphs.glyphSet() if self.componentGlyphs else tuple()
        builder.add_glyphClassDef(self.location, base, liga, mark, comp)

    def asFea(self, indent=''):
        return 'GlyphClassDef {}, {}, {}, {};'.format(self.baseGlyphs.asFea() if self.baseGlyphs else '', self.ligatureGlyphs.asFea() if self.ligatureGlyphs else '', self.markGlyphs.asFea() if self.markGlyphs else '', self.componentGlyphs.asFea() if self.componentGlyphs else '')