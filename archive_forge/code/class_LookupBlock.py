from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LookupBlock(Block):
    """A named lookup, containing ``statements``."""

    def __init__(self, name, use_extension=False, location=None):
        Block.__init__(self, location)
        self.name, self.use_extension = (name, use_extension)

    def build(self, builder):
        builder.start_lookup_block(self.location, self.name)
        Block.build(self, builder)
        builder.end_lookup_block()

    def asFea(self, indent=''):
        res = 'lookup {} '.format(self.name)
        if self.use_extension:
            res += 'useExtension '
        res += '{\n'
        res += Block.asFea(self, indent=indent)
        res += '{}}} {};\n'.format(indent, self.name)
        return res