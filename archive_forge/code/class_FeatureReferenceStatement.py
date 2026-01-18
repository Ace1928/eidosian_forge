from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class FeatureReferenceStatement(Statement):
    """Example: ``feature salt;``"""

    def __init__(self, featureName, location=None):
        Statement.__init__(self, location)
        self.location, self.featureName = (location, featureName)

    def build(self, builder):
        """Calls the builder object's ``add_feature_reference`` callback."""
        builder.add_feature_reference(self.location, self.featureName)

    def asFea(self, indent=''):
        return 'feature {};'.format(self.featureName)