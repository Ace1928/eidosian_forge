from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class IgnoreSubstStatement(Statement):
    """An ``ignore sub`` statement, containing `one or more` contexts to ignore.

    ``chainContexts`` should be a list of ``(prefix, glyphs, suffix)`` tuples,
    with each of ``prefix``, ``glyphs`` and ``suffix`` being
    `glyph-containing objects`_ ."""

    def __init__(self, chainContexts, location=None):
        Statement.__init__(self, location)
        self.chainContexts = chainContexts

    def build(self, builder):
        """Calls the builder object's ``add_chain_context_subst`` callback on
        each rule context."""
        for prefix, glyphs, suffix in self.chainContexts:
            prefix = [p.glyphSet() for p in prefix]
            glyphs = [g.glyphSet() for g in glyphs]
            suffix = [s.glyphSet() for s in suffix]
            builder.add_chain_context_subst(self.location, prefix, glyphs, suffix, [])

    def asFea(self, indent=''):
        contexts = []
        for prefix, glyphs, suffix in self.chainContexts:
            res = ''
            if len(prefix):
                res += ' '.join(map(asFea, prefix)) + ' '
            res += ' '.join((g.asFea() + "'" for g in glyphs))
            if len(suffix):
                res += ' ' + ' '.join(map(asFea, suffix))
            contexts.append(res)
        return 'ignore sub ' + ', '.join(contexts) + ';'