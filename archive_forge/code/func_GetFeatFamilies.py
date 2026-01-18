import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def GetFeatFamilies(self):
    fams = [fam for fam in self.featFactory.GetFeatureFamilies() if fam not in self.skipFeats]
    fams.sort()
    return fams