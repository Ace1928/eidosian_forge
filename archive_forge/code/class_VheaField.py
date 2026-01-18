from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class VheaField(Statement):
    """An entry in the ``vhea`` table."""

    def __init__(self, key, value, location=None):
        Statement.__init__(self, location)
        self.key = key
        self.value = value

    def build(self, builder):
        """Calls the builder object's ``add_vhea_field`` callback."""
        builder.add_vhea_field(self.key, self.value)

    def asFea(self, indent=''):
        fields = ('VertTypoAscender', 'VertTypoDescender', 'VertTypoLineGap')
        keywords = dict([(x.lower(), x) for x in fields])
        return '{} {};'.format(keywords[self.key], self.value)