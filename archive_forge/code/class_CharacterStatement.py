from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class CharacterStatement(Statement):
    """
    Statement used in cvParameters blocks of Character Variant features (cvXX).
    The Unicode value may be written with either decimal or hexadecimal
    notation. The value must be preceded by '0x' if it is a hexadecimal value.
    The largest Unicode value allowed is 0xFFFFFF.
    """

    def __init__(self, character, tag, location=None):
        Statement.__init__(self, location)
        self.character = character
        self.tag = tag

    def build(self, builder):
        """Calls the builder object's ``add_cv_character`` callback."""
        builder.add_cv_character(self.character, self.tag)

    def asFea(self, indent=''):
        return 'Character {:#x};'.format(self.character)