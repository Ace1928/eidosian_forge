from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LookupReferenceStatement(Statement):
    """Represents a ``lookup ...;`` statement to include a lookup in a feature.

    The ``lookup`` should be a :class:`LookupBlock` object."""

    def __init__(self, lookup, location=None):
        Statement.__init__(self, location)
        self.location, self.lookup = (location, lookup)

    def build(self, builder):
        """Calls the builder object's ``add_lookup_call`` callback."""
        builder.add_lookup_call(self.lookup.name)

    def asFea(self, indent=''):
        return 'lookup {};'.format(self.lookup.name)