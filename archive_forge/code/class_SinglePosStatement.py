from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class SinglePosStatement(Statement):
    """A single position statement. ``prefix`` and ``suffix`` should be
    lists of `glyph-containing objects`_.

    ``pos`` should be a one-element list containing a (`glyph-containing object`_,
    :class:`ValueRecord`) tuple."""

    def __init__(self, pos, prefix, suffix, forceChain, location=None):
        Statement.__init__(self, location)
        self.pos, self.prefix, self.suffix = (pos, prefix, suffix)
        self.forceChain = forceChain

    def build(self, builder):
        """Calls the builder object's ``add_single_pos`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        pos = [(g.glyphSet(), value) for g, value in self.pos]
        builder.add_single_pos(self.location, prefix, suffix, pos, self.forceChain)

    def asFea(self, indent=''):
        res = 'pos '
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += ' '.join(map(asFea, self.prefix)) + ' '
            res += ' '.join([asFea(x[0]) + "'" + (' ' + x[1].asFea() if x[1] else '') for x in self.pos])
            if len(self.suffix):
                res += ' ' + ' '.join(map(asFea, self.suffix))
        else:
            res += ' '.join([asFea(x[0]) + ' ' + (x[1].asFea() if x[1] else '') for x in self.pos])
        res += ';'
        return res