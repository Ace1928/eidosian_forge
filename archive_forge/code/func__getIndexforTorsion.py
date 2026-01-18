import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _getIndexforTorsion(neighbors, inv):
    """ Helper function to calculate the index of the reference atom for 
      a given atom

      Arguments:
      - neighbors:  list of the neighbors of the atom
      - inv:        atom invariants

      Return: list of atom indices as reference for torsion
  """
    if len(neighbors) == 1:
        return [neighbors[0]]
    elif _doMatch(inv, neighbors):
        return neighbors
    elif _doNotMatch(inv, neighbors):
        neighbors = sorted(neighbors, key=lambda x: inv[x.GetIdx()])
        return [neighbors[0]]
    elif len(neighbors) == 3:
        at = _doMatchExcept1(inv, neighbors)
        if at is None:
            raise ValueError('Atom neighbors are either all the same or all different')
        return [at]
    else:
        neighbors = sorted(neighbors, key=lambda x: inv[x.GetIdx()])
        return [neighbors[0]]