from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class MultipleSubstStatement(Statement):
    """A multiple substitution statement.

    Args:
        prefix: a list of `glyph-containing objects`_.
        glyph: a single glyph-containing object.
        suffix: a list of glyph-containing objects.
        replacement: a list of glyph-containing objects.
        forceChain: If true, the statement is expressed as a chaining rule
            (e.g. ``sub f' i' by f_i``) even when no context is given.
    """

    def __init__(self, prefix, glyph, suffix, replacement, forceChain=False, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyph, self.suffix = (prefix, glyph, suffix)
        self.replacement = replacement
        self.forceChain = forceChain

    def build(self, builder):
        """Calls the builder object's ``add_multiple_subst`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        if hasattr(self.glyph, 'glyphSet'):
            originals = self.glyph.glyphSet()
        else:
            originals = [self.glyph]
        count = len(originals)
        replaces = []
        for r in self.replacement:
            if hasattr(r, 'glyphSet'):
                replace = r.glyphSet()
            else:
                replace = [r]
            if len(replace) == 1 and len(replace) != count:
                replace = replace * count
            replaces.append(replace)
        replaces = list(zip(*replaces))
        seen_originals = set()
        for i, original in enumerate(originals):
            if original not in seen_originals:
                seen_originals.add(original)
                builder.add_multiple_subst(self.location, prefix, original, suffix, replaces and replaces[i] or (), self.forceChain)

    def asFea(self, indent=''):
        res = 'sub '
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += ' '.join(map(asFea, self.prefix)) + ' '
            res += asFea(self.glyph) + "'"
            if len(self.suffix):
                res += ' ' + ' '.join(map(asFea, self.suffix))
        else:
            res += asFea(self.glyph)
        replacement = self.replacement or [NullGlyph()]
        res += ' by '
        res += ' '.join(map(asFea, replacement))
        res += ';'
        return res