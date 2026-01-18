from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ConditionsetStatement(Statement):
    """
    A variable layout conditionset

    Args:
        name (str): the name of this conditionset
        conditions (dict): a dictionary mapping axis tags to a
            tuple of (min,max) userspace coordinates.
    """

    def __init__(self, name, conditions, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.conditions = conditions

    def build(self, builder):
        builder.add_conditionset(self.location, self.name, self.conditions)

    def asFea(self, res='', indent=''):
        res += indent + f'conditionset {self.name} ' + '{\n'
        for tag, (minvalue, maxvalue) in self.conditions.items():
            res += indent + SHIFT + f'{tag} {minvalue} {maxvalue};\n'
        res += indent + '}' + f' {self.name};\n'
        return res