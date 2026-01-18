from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class STATDesignAxisStatement(Statement):
    """A STAT table Design Axis

    Args:
        tag (str): a 4 letter axis tag
        axisOrder (int): an int
        names (list): a list of :class:`STATNameStatement` objects
    """

    def __init__(self, tag, axisOrder, names, location=None):
        Statement.__init__(self, location)
        self.tag = tag
        self.axisOrder = axisOrder
        self.names = names
        self.location = location

    def build(self, builder):
        builder.addDesignAxis(self, self.location)

    def asFea(self, indent=''):
        indent += SHIFT
        res = f'DesignAxis {self.tag} {self.axisOrder} {{ \n'
        res += ('\n' + indent).join([s.asFea(indent=indent) for s in self.names]) + '\n'
        res += '};'
        return res