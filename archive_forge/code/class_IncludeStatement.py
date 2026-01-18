from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class IncludeStatement(Statement):
    """An ``include()`` statement."""

    def __init__(self, filename, location=None):
        super(IncludeStatement, self).__init__(location)
        self.filename = filename

    def build(self):
        raise FeatureLibError('Building an include statement is not implemented yet. Instead, use Parser(..., followIncludes=True) for building.', self.location)

    def asFea(self, indent=''):
        return indent + 'include(%s);' % self.filename