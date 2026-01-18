from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LigatureSubstStatement(Statement):
    """A chained contextual substitution statement.

    ``prefix``, ``glyphs``, and ``suffix`` should be lists of
    `glyph-containing objects`_; ``replacement`` should be a single
    `glyph-containing object`_.

    If ``forceChain`` is True, this is expressed as a chaining rule
    (e.g. ``sub f' i' by f_i``) even when no context is given."""

    def __init__(self, prefix, glyphs, suffix, replacement, forceChain, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyphs, self.suffix = (prefix, glyphs, suffix)
        self.replacement, self.forceChain = (replacement, forceChain)

    def build(self, builder):
        prefix = [p.glyphSet() for p in self.prefix]
        glyphs = [g.glyphSet() for g in self.glyphs]
        suffix = [s.glyphSet() for s in self.suffix]
        builder.add_ligature_subst(self.location, prefix, glyphs, suffix, self.replacement, self.forceChain)

    def asFea(self, indent=''):
        res = 'sub '
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += ' '.join((g.asFea() for g in self.prefix)) + ' '
            res += ' '.join((g.asFea() + "'" for g in self.glyphs))
            if len(self.suffix):
                res += ' ' + ' '.join((g.asFea() for g in self.suffix))
        else:
            res += ' '.join((g.asFea() for g in self.glyphs))
        res += ' by '
        res += asFea(self.replacement)
        res += ';'
        return res