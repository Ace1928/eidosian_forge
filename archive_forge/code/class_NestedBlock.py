from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class NestedBlock(Block):
    """A block inside another block, for example when found inside a
    ``cvParameters`` block."""

    def __init__(self, tag, block_name, location=None):
        Block.__init__(self, location)
        self.tag = tag
        self.block_name = block_name

    def build(self, builder):
        Block.build(self, builder)
        if self.block_name == 'ParamUILabelNameID':
            builder.add_to_cv_num_named_params(self.tag)

    def asFea(self, indent=''):
        res = '{}{} {{\n'.format(indent, self.block_name)
        res += Block.asFea(self, indent=indent)
        res += '{}}};\n'.format(indent)
        return res