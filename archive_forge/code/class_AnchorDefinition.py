from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class AnchorDefinition(Statement):
    """A named anchor definition. (2.e.viii). ``name`` should be a string."""

    def __init__(self, name, x, y, contourpoint=None, location=None):
        Statement.__init__(self, location)
        self.name, self.x, self.y, self.contourpoint = (name, x, y, contourpoint)

    def asFea(self, indent=''):
        res = 'anchorDef {} {}'.format(self.x, self.y)
        if self.contourpoint:
            res += ' contourpoint {}'.format(self.contourpoint)
        res += ' {};'.format(self.name)
        return res