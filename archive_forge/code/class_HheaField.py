from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class HheaField(Statement):
    """An entry in the ``hhea`` table."""

    def __init__(self, key, value, location=None):
        Statement.__init__(self, location)
        self.key = key
        self.value = value

    def build(self, builder):
        """Calls the builder object's ``add_hhea_field`` callback."""
        builder.add_hhea_field(self.key, self.value)

    def asFea(self, indent=''):
        fields = ('CaretOffset', 'Ascender', 'Descender', 'LineGap')
        keywords = dict([(x.lower(), x) for x in fields])
        return '{} {};'.format(keywords[self.key], self.value)