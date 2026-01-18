from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class CursivePosStatement(Statement):
    """A cursive positioning statement. Entry and exit anchors can either
    be :class:`Anchor` objects or ``None``."""

    def __init__(self, glyphclass, entryAnchor, exitAnchor, location=None):
        Statement.__init__(self, location)
        self.glyphclass = glyphclass
        self.entryAnchor, self.exitAnchor = (entryAnchor, exitAnchor)

    def build(self, builder):
        """Calls the builder object's ``add_cursive_pos`` callback."""
        builder.add_cursive_pos(self.location, self.glyphclass.glyphSet(), self.entryAnchor, self.exitAnchor)

    def asFea(self, indent=''):
        entry = self.entryAnchor.asFea() if self.entryAnchor else '<anchor NULL>'
        exit = self.exitAnchor.asFea() if self.exitAnchor else '<anchor NULL>'
        return 'pos cursive {} {} {};'.format(self.glyphclass.asFea(), entry, exit)