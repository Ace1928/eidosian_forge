from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class FeatureFile(Block):
    """The top-level element of the syntax tree, containing the whole feature
    file in its ``statements`` attribute."""

    def __init__(self):
        Block.__init__(self, location=None)
        self.markClasses = {}

    def asFea(self, indent=''):
        return '\n'.join((s.asFea(indent=indent) for s in self.statements))