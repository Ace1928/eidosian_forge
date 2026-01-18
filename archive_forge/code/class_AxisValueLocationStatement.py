from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class AxisValueLocationStatement(Statement):
    """
    A STAT table Axis Value Location

    Args:
        tag (str): a 4 letter axis tag
        values (list): a list of ints and/or floats
    """

    def __init__(self, tag, values, location=None):
        Statement.__init__(self, location)
        self.tag = tag
        self.values = values

    def asFea(self, res=''):
        res += f'location {self.tag} '
        res += f'{' '.join((str(i) for i in self.values))};\n'
        return res