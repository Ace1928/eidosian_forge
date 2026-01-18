import math
import numpy
from rdkit import Chem, Geometry
def GetDonor2FeatVects(conf, featAtoms, scale=1.5):
    """
  Get the direction vectors for Donor of type 2

  This is a donor with two heavy atoms as neighbors. The atom may are may not have
  hydrogen on it. Here are the situations with the neighbors that will be considered here
    1. two heavy atoms and two hydrogens: we will assume a sp3 arrangement here
    2. two heavy atoms and one hydrogen: this can either be sp2 or sp3
    3. two heavy atoms and no hydrogens
    
  ARGUMENTS:
    featAtoms - list of atoms that are part of the feature
    scale - length of the direction vector
  """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    cpt = conf.GetAtomPosition(aid)
    nbrs = list(mol.GetAtomWithIdx(aid).GetNeighbors())
    assert len(nbrs) >= 2
    hydrogens = []
    heavy = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hydrogens.append(nbr)
        else:
            heavy.append(nbr)
    if len(nbrs) == 2:
        assert len(hydrogens) == 0
        bvec = _findAvgVec(conf, cpt, heavy)
        bvec *= -1.0 * scale
        bvec += cpt
        return (((cpt, bvec),), 'linear')
    if len(nbrs) == 3:
        assert len(hydrogens) == 1
        hid = hydrogens[0].GetIdx()
        bvec = conf.GetAtomPosition(hid)
        bvec -= cpt
        bvec.Normalize()
        bvec *= scale
        bvec += cpt
        if _checkPlanarity(conf, cpt, nbrs, tol=0.01):
            return (((cpt, bvec),), 'linear')
        ovec = _findAvgVec(conf, cpt, heavy)
        ovec *= -1.0 * scale
        ovec += cpt
        return (((cpt, bvec), (cpt, ovec)), 'linear')
    if len(nbrs) >= 4:
        res = []
        for hid in hydrogens:
            hid = hid.GetIdx()
            bvec = conf.GetAtomPosition(hid)
            bvec -= cpt
            bvec.Normalize()
            bvec *= scale
            bvec += cpt
            res.append((cpt, bvec))
        return (tuple(res), 'linear')
    return None