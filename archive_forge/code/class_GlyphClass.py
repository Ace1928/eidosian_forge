from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class GlyphClass(Expression):
    """A glyph class, such as ``[acute cedilla grave]``."""

    def __init__(self, glyphs=None, location=None):
        Expression.__init__(self, location)
        self.glyphs = glyphs if glyphs is not None else []
        self.original = []
        self.curr = 0

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphs)

    def asFea(self, indent=''):
        if len(self.original):
            if self.curr < len(self.glyphs):
                self.original.extend(self.glyphs[self.curr:])
                self.curr = len(self.glyphs)
            return '[' + ' '.join(map(asFea, self.original)) + ']'
        else:
            return '[' + ' '.join(map(asFea, self.glyphs)) + ']'

    def extend(self, glyphs):
        """Add a list of :class:`GlyphName` objects to the class."""
        self.glyphs.extend(glyphs)

    def append(self, glyph):
        """Add a single :class:`GlyphName` object to the class."""
        self.glyphs.append(glyph)

    def add_range(self, start, end, glyphs):
        """Add a range (e.g. ``A-Z``) to the class. ``start`` and ``end``
        are either :class:`GlyphName` objects or strings representing the
        start and end glyphs in the class, and ``glyphs`` is the full list of
        :class:`GlyphName` objects in the range."""
        if self.curr < len(self.glyphs):
            self.original.extend(self.glyphs[self.curr:])
        self.original.append((start, end))
        self.glyphs.extend(glyphs)
        self.curr = len(self.glyphs)

    def add_cid_range(self, start, end, glyphs):
        """Add a range to the class by glyph ID. ``start`` and ``end`` are the
        initial and final IDs, and ``glyphs`` is the full list of
        :class:`GlyphName` objects in the range."""
        if self.curr < len(self.glyphs):
            self.original.extend(self.glyphs[self.curr:])
        self.original.append(('\\{}'.format(start), '\\{}'.format(end)))
        self.glyphs.extend(glyphs)
        self.curr = len(self.glyphs)

    def add_class(self, gc):
        """Add glyphs from the given :class:`GlyphClassName` object to the
        class."""
        if self.curr < len(self.glyphs):
            self.original.extend(self.glyphs[self.curr:])
        self.original.append(gc)
        self.glyphs.extend(gc.glyphSet())
        self.curr = len(self.glyphs)