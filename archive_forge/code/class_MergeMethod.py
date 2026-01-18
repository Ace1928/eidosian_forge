import copy
from rdkit.Chem.FeatMaps import FeatMaps
class MergeMethod(object):
    WeightedAverage = 0
    Average = 1
    UseLarger = 2

    @classmethod
    def valid(cls, mergeMethod):
        """ Check that mergeMethod is valid """
        if mergeMethod not in (cls.WeightedAverage, cls.Average, cls.UseLarger):
            raise ValueError('unrecognized mergeMethod')