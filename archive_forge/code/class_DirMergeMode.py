import copy
from rdkit.Chem.FeatMaps import FeatMaps
class DirMergeMode(object):
    NoMerge = 0
    Sum = 1

    @classmethod
    def valid(cls, dirMergeMode):
        """ Check that dirMergeMode is valid """
        if dirMergeMode not in (cls.NoMerge, cls.Sum):
            raise ValueError('unrecognized dirMergeMode')