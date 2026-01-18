from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class FeatureNameStatement(NameRecord):
    """Represents a ``sizemenuname`` or ``name`` statement."""

    def build(self, builder):
        """Calls the builder object's ``add_featureName`` callback."""
        NameRecord.build(self, builder)
        builder.add_featureName(self.nameID)

    def asFea(self, indent=''):
        if self.nameID == 'size':
            tag = 'sizemenuname'
        else:
            tag = 'name'
        plat = simplify_name_attributes(self.platformID, self.platEncID, self.langID)
        if plat != '':
            plat += ' '
        return '{} {}"{}";'.format(tag, plat, self.string)