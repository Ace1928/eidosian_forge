from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class MarkLigPosStatement(Statement):
    """A mark-to-ligature positioning rule. The ``ligatures`` must be a
    `glyph-containing object`_. The ``marks`` should be a list of lists: each
    element in the top-level list represents a component glyph, and is made
    up of a list of (:class:`Anchor`, :class:`MarkClass`) tuples representing
    mark attachment points for that position.

    Example::

        m1 = MarkClass("TOP_MARKS")
        m2 = MarkClass("BOTTOM_MARKS")
        # ... add definitions to mark classes...

        glyph = GlyphName("lam_meem_jeem")
        marks = [
            [ (Anchor(625,1800), m1) ], # Attachments on 1st component (lam)
            [ (Anchor(376,-378), m2) ], # Attachments on 2nd component (meem)
            [ ]                         # No attachments on the jeem
        ]
        mlp = MarkLigPosStatement(glyph, marks)

        mlp.asFea()
        # pos ligature lam_meem_jeem <anchor 625 1800> mark @TOP_MARKS
        # ligComponent <anchor 376 -378> mark @BOTTOM_MARKS;

    """

    def __init__(self, ligatures, marks, location=None):
        Statement.__init__(self, location)
        self.ligatures, self.marks = (ligatures, marks)

    def build(self, builder):
        """Calls the builder object's ``add_mark_lig_pos`` callback."""
        builder.add_mark_lig_pos(self.location, self.ligatures.glyphSet(), self.marks)

    def asFea(self, indent=''):
        res = 'pos ligature {}'.format(self.ligatures.asFea())
        ligs = []
        for l in self.marks:
            temp = ''
            if l is None or not len(l):
                temp = '\n' + indent + SHIFT * 2 + '<anchor NULL>'
            else:
                for a, m in l:
                    temp += '\n' + indent + SHIFT * 2 + '{} mark @{}'.format(a.asFea(), m.name)
            ligs.append(temp)
        res += ('\n' + indent + SHIFT + 'ligComponent').join(ligs)
        res += ';'
        return res