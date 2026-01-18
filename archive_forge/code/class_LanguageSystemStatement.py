from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class LanguageSystemStatement(Statement):
    """A top-level ``languagesystem`` statement."""

    def __init__(self, script, language, location=None):
        Statement.__init__(self, location)
        self.script, self.language = (script, language)

    def build(self, builder):
        """Calls the builder object's ``add_language_system`` callback."""
        builder.add_language_system(self.location, self.script, self.language)

    def asFea(self, indent=''):
        return 'languagesystem {} {};'.format(self.script, self.language.strip())