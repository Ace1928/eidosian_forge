from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class STATAxisValueStatement(Statement):
    """A STAT table Axis Value Record

    Args:
        names (list): a list of :class:`STATNameStatement` objects
        locations (list): a list of :class:`AxisValueLocationStatement` objects
        flags (int): an int
    """

    def __init__(self, names, locations, flags, location=None):
        Statement.__init__(self, location)
        self.names = names
        self.locations = locations
        self.flags = flags

    def build(self, builder):
        builder.addAxisValueRecord(self, self.location)

    def asFea(self, indent=''):
        res = 'AxisValue {\n'
        for location in self.locations:
            res += location.asFea()
        for nameRecord in self.names:
            res += nameRecord.asFea()
            res += '\n'
        if self.flags:
            flags = ['OlderSiblingFontAttribute', 'ElidableAxisValueName']
            flagStrings = []
            curr = 1
            for i in range(len(flags)):
                if self.flags & curr != 0:
                    flagStrings.append(flags[i])
                curr = curr << 1
            res += f'flag {' '.join(flagStrings)};\n'
        res += '};'
        return res