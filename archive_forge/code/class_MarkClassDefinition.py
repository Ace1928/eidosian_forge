from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class MarkClassDefinition(Statement):
    """A single ``markClass`` statement. The ``markClass`` should be a
    :class:`MarkClass` object, the ``anchor`` an :class:`Anchor` object,
    and the ``glyphs`` parameter should be a `glyph-containing object`_ .

    Example:

        .. code:: python

            mc = MarkClass("FRENCH_ACCENTS")
            mc.addDefinition( MarkClassDefinition(mc, Anchor(350, 800),
                GlyphClass([ GlyphName("acute"), GlyphName("grave") ])
            ) )
            mc.addDefinition( MarkClassDefinition(mc, Anchor(350, -200),
                GlyphClass([ GlyphName("cedilla") ])
            ) )

            mc.asFea()
            # markClass [acute grave] <anchor 350 800> @FRENCH_ACCENTS;
            # markClass [cedilla] <anchor 350 -200> @FRENCH_ACCENTS;

    """

    def __init__(self, markClass, anchor, glyphs, location=None):
        Statement.__init__(self, location)
        assert isinstance(markClass, MarkClass)
        assert isinstance(anchor, Anchor) and isinstance(glyphs, Expression)
        self.markClass, self.anchor, self.glyphs = (markClass, anchor, glyphs)

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return self.glyphs.glyphSet()

    def asFea(self, indent=''):
        return 'markClass {} {} @{};'.format(self.glyphs.asFea(), self.anchor.asFea(), self.markClass.name)