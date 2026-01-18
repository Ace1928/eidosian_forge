import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def GetMolFeats(self, mol):
    featFamilies = self.GetFeatFamilies()
    featMatches = {}
    for fam in featFamilies:
        feats = self.featFactory.GetFeaturesForMol(mol, includeOnly=fam)
        featMatches[fam] = [feat.GetAtomIds() for feat in feats]
    return [featMatches[x] for x in featFamilies]