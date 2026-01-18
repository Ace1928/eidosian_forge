from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ChainContextSubstStatement(Statement):
    """A chained contextual substitution statement.

    ``prefix``, ``glyphs``, and ``suffix`` should be lists of
    `glyph-containing objects`_ .

    ``lookups`` should be a list of elements representing what lookups
    to apply at each glyph position. Each element should be a
    :class:`LookupBlock` to apply a single chaining lookup at the given
    position, a list of :class:`LookupBlock`\\ s to apply multiple
    lookups, or ``None`` to apply no lookup. The length of the outer
    list should equal the length of ``glyphs``; the inner lists can be
    of variable length."""

    def __init__(self, prefix, glyphs, suffix, lookups, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyphs, self.suffix = (prefix, glyphs, suffix)
        self.lookups = list(lookups)
        for i, lookup in enumerate(lookups):
            if lookup:
                try:
                    (_ for _ in lookup)
                except TypeError:
                    self.lookups[i] = [lookup]

    def build(self, builder):
        """Calls the builder's ``add_chain_context_subst`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        glyphs = [g.glyphSet() for g in self.glyphs]
        suffix = [s.glyphSet() for s in self.suffix]
        builder.add_chain_context_subst(self.location, prefix, glyphs, suffix, self.lookups)

    def asFea(self, indent=''):
        res = 'sub '
        if len(self.prefix) or len(self.suffix) or any([x is not None for x in self.lookups]):
            if len(self.prefix):
                res += ' '.join((g.asFea() for g in self.prefix)) + ' '
            for i, g in enumerate(self.glyphs):
                res += g.asFea() + "'"
                if self.lookups[i]:
                    for lu in self.lookups[i]:
                        res += ' lookup ' + lu.name
                if i < len(self.glyphs) - 1:
                    res += ' '
            if len(self.suffix):
                res += ' ' + ' '.join(map(asFea, self.suffix))
        else:
            res += ' '.join(map(asFea, self.glyph))
        res += ';'
        return res