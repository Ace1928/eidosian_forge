import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _doMatchExcept1(inv, atoms):
    """ Helper function to check if two atoms in the list are the same, 
      and one not
      Note: Works only for three atoms
      
      Arguments:
      - inv:    atom invariants (used to define equivalence of atoms)
      - atoms:  list of atoms to check

      Return: atom that is different
  """
    if len(atoms) != 3:
        raise ValueError('Number of atoms must be three')
    a1 = atoms[0].GetIdx()
    a2 = atoms[1].GetIdx()
    a3 = atoms[2].GetIdx()
    if inv[a1] == inv[a2] and inv[a1] != inv[a3] and (inv[a2] != inv[a3]):
        return atoms[2]
    elif inv[a1] != inv[a2] and inv[a1] == inv[a3] and (inv[a2] != inv[a3]):
        return atoms[1]
    elif inv[a1] != inv[a2] and inv[a1] != inv[a3] and (inv[a2] == inv[a3]):
        return atoms[0]
    return None