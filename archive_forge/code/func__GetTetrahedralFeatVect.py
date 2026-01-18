import math
import numpy
from rdkit import Chem, Geometry
def _GetTetrahedralFeatVect(conf, aid, scale):
    mol = conf.GetOwningMol()
    cpt = conf.GetAtomPosition(aid)
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
    if not _checkPlanarity(conf, cpt, nbrs, tol=0.1):
        bvec = _findAvgVec(conf, cpt, nbrs)
        bvec *= -1.0 * scale
        bvec += cpt
        return ((cpt, bvec),)
    return tuple()