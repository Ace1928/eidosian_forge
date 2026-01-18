from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class CVParametersNameStatement(NameRecord):
    """Represent a name statement inside a ``cvParameters`` block."""

    def __init__(self, nameID, platformID, platEncID, langID, string, block_name, location=None):
        NameRecord.__init__(self, nameID, platformID, platEncID, langID, string, location=location)
        self.block_name = block_name

    def build(self, builder):
        """Calls the builder object's ``add_cv_parameter`` callback."""
        item = ''
        if self.block_name == 'ParamUILabelNameID':
            item = '_{}'.format(builder.cv_num_named_params_.get(self.nameID, 0))
        builder.add_cv_parameter(self.nameID)
        self.nameID = (self.nameID, self.block_name + item)
        NameRecord.build(self, builder)

    def asFea(self, indent=''):
        plat = simplify_name_attributes(self.platformID, self.platEncID, self.langID)
        if plat != '':
            plat += ' '
        return 'name {}"{}";'.format(plat, self.string)