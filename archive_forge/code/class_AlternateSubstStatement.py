from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class AlternateSubstStatement(Statement):
    """A ``sub ... from ...`` statement.

    ``prefix``, ``glyph``, ``suffix`` and ``replacement`` should be lists of
    `glyph-containing objects`_. ``glyph`` should be a `one element list`."""

    def __init__(self, prefix, glyph, suffix, replacement, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyph, self.suffix = (prefix, glyph, suffix)
        self.replacement = replacement

    def build(self, builder):
        """Calls the builder's ``add_alternate_subst`` callback."""
        glyph = self.glyph.glyphSet()
        assert len(glyph) == 1, glyph
        glyph = list(glyph)[0]
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        replacement = self.replacement.glyphSet()
        builder.add_alternate_subst(self.location, prefix, glyph, suffix, replacement)

    def asFea(self, indent=''):
        res = 'sub '
        if len(self.prefix) or len(self.suffix):
            if len(self.prefix):
                res += ' '.join(map(asFea, self.prefix)) + ' '
            res += asFea(self.glyph) + "'"
            if len(self.suffix):
                res += ' ' + ' '.join(map(asFea, self.suffix))
        else:
            res += asFea(self.glyph)
        res += ' from '
        res += asFea(self.replacement)
        res += ';'
        return res