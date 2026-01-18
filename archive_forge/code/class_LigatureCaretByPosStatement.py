from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LigatureCaretByPosStatement(Statement):
    """A ``GDEF`` table ``LigatureCaretByPos`` statement. ``glyphs`` should be
    a `glyph-containing object`_, and ``carets`` should be a list of integers."""

    def __init__(self, glyphs, carets, location=None):
        Statement.__init__(self, location)
        self.glyphs, self.carets = (glyphs, carets)

    def build(self, builder):
        """Calls the builder object's ``add_ligatureCaretByPos_`` callback."""
        glyphs = self.glyphs.glyphSet()
        builder.add_ligatureCaretByPos_(self.location, glyphs, set(self.carets))

    def asFea(self, indent=''):
        return 'LigatureCaretByPos {} {};'.format(self.glyphs.asFea(), ' '.join((str(x) for x in self.carets)))