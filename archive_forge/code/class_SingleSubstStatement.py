from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class SingleSubstStatement(Statement):
    """A single substitution statement.

    Note the unusual argument order: ``prefix`` and suffix come `after`
    the replacement ``glyphs``. ``prefix``, ``suffix``, ``glyphs`` and
    ``replace`` should be lists of `glyph-containing objects`_. ``glyphs`` and
    ``replace`` should be one-item lists.
    """

    def __init__(self, glyphs, replace, prefix, suffix, forceChain, location=None):
        Statement.__init__(self, location)
        self.prefix, self.suffix = (prefix, suffix)
        self.forceChain = forceChain
        self.glyphs = glyphs
        self.replacements = replace

    def build(self, builder):
        """Calls the builder object's ``add_single_subst`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        originals = self.glyphs[0].glyphSet()
        replaces = self.replacements[0].glyphSet()
        if len(replaces) == 1:
            replaces = replaces * len(originals)
        builder.add_single_subst(self.location, prefix, suffix, OrderedDict(zip(originals, replaces)), self.forceChain)

    def asFea(self, indent=''):
        res = 'sub '
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += ' '.join((asFea(g) for g in self.prefix)) + ' '
            res += ' '.join((asFea(g) + "'" for g in self.glyphs))
            if len(self.suffix):
                res += ' ' + ' '.join((asFea(g) for g in self.suffix))
        else:
            res += ' '.join((asFea(g) for g in self.glyphs))
        res += ' by {};'.format(' '.join((asFea(g) for g in self.replacements)))
        return res