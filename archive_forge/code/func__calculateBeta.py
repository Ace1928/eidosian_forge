import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _calculateBeta(mol, distmat, aid1):
    """ Helper function to calculate the beta for torsion weights
      according to the formula in the paper.
      w(dmax/2) = 0.1

      Arguments:
      - mol:     the molecule of interest
      - distmat: distance matrix of the molecule
      - aid1:    atom index of the most central atom

      Return: value of beta (float)
  """
    bonds = []
    for b in mol.GetBonds():
        nb1 = _getHeavyAtomNeighbors(b.GetBeginAtom())
        nb2 = _getHeavyAtomNeighbors(b.GetEndAtom())
        if len(nb2) > 1 and len(nb2) > 1:
            bonds.append(b)
    dmax = 0
    for b in bonds:
        bid1 = b.GetBeginAtom().GetIdx()
        bid2 = b.GetEndAtom().GetIdx()
        d = max([distmat[aid1][bid1], distmat[aid1][bid2]])
        if d > dmax:
            dmax = d
    dmax2 = dmax / 2.0
    beta = -math.log(0.1) / (dmax2 * dmax2)
    return beta