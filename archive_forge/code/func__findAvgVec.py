import math
import numpy
from rdkit import Chem, Geometry
def _findAvgVec(conf, center, nbrs):
    avgVec = 0
    for nbr in nbrs:
        nid = nbr.GetIdx()
        pt = conf.GetAtomPosition(nid)
        pt -= center
        pt.Normalize()
        if avgVec == 0:
            avgVec = pt
        else:
            avgVec += pt
    avgVec.Normalize()
    return avgVec