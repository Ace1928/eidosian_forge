import math
import numpy
from rdkit import Chem, Geometry
def _checkPlanarity(conf, cpt, nbrs, tol=0.001):
    assert len(nbrs) == 3
    v1 = conf.GetAtomPosition(nbrs[0].GetIdx())
    v1 -= cpt
    v2 = conf.GetAtomPosition(nbrs[1].GetIdx())
    v2 -= cpt
    v3 = conf.GetAtomPosition(nbrs[2].GetIdx())
    v3 -= cpt
    normal = v1.CrossProduct(v2)
    dotP = abs(v3.DotProduct(normal))
    return int(dotP <= tol)