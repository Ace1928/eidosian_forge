from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class VariationBlock(Block):
    """A variation feature block, applicable in a given set of conditions."""

    def __init__(self, name, conditionset, use_extension=False, location=None):
        Block.__init__(self, location)
        self.name, self.conditionset, self.use_extension = (name, conditionset, use_extension)

    def build(self, builder):
        """Call the ``start_feature`` callback on the builder object, visit
        all the statements in this feature, and then call ``end_feature``."""
        builder.start_feature(self.location, self.name)
        if self.conditionset != 'NULL' and self.conditionset not in builder.conditionsets_:
            raise FeatureLibError(f'variation block used undefined conditionset {self.conditionset}', self.location)
        features = builder.features_
        builder.features_ = {}
        Block.build(self, builder)
        for key, value in builder.features_.items():
            items = builder.feature_variations_.setdefault(key, {}).setdefault(self.conditionset, [])
            items.extend(value)
            if key not in features:
                features[key] = []
        builder.features_ = features
        builder.end_feature()

    def asFea(self, indent=''):
        res = indent + 'variation %s ' % self.name.strip()
        res += self.conditionset + ' '
        if self.use_extension:
            res += 'useExtension '
        res += '{\n'
        res += Block.asFea(self, indent=indent)
        res += indent + '} %s;\n' % self.name.strip()
        return res