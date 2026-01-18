from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LookupFlagStatement(Statement):
    """A ``lookupflag`` statement. The ``value`` should be an integer value
    representing the flags in use, but not including the ``markAttachment``
    class and ``markFilteringSet`` values, which must be specified as
    glyph-containing objects."""

    def __init__(self, value=0, markAttachment=None, markFilteringSet=None, location=None):
        Statement.__init__(self, location)
        self.value = value
        self.markAttachment = markAttachment
        self.markFilteringSet = markFilteringSet

    def build(self, builder):
        """Calls the builder object's ``set_lookup_flag`` callback."""
        markAttach = None
        if self.markAttachment is not None:
            markAttach = self.markAttachment.glyphSet()
        markFilter = None
        if self.markFilteringSet is not None:
            markFilter = self.markFilteringSet.glyphSet()
        builder.set_lookup_flag(self.location, self.value, markAttach, markFilter)

    def asFea(self, indent=''):
        res = []
        flags = ['RightToLeft', 'IgnoreBaseGlyphs', 'IgnoreLigatures', 'IgnoreMarks']
        curr = 1
        for i in range(len(flags)):
            if self.value & curr != 0:
                res.append(flags[i])
            curr = curr << 1
        if self.markAttachment is not None:
            res.append('MarkAttachmentType {}'.format(self.markAttachment.asFea()))
        if self.markFilteringSet is not None:
            res.append('UseMarkFilteringSet {}'.format(self.markFilteringSet.asFea()))
        if not res:
            res = ['0']
        return 'lookupflag {};'.format(' '.join(res))